"""Harness for API-level integration tests: the real app on the test containers, plus builders
for users, organizations and tokens. Rows are committed (the app uses its own transactions), so
every builder uses unique phones, e-mails and names."""

import io
import os
import re
import secrets
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from medlens.adapters.fakes import (
    AsyncFakeLLMProvider,
    FakeBarcode,
    FakeImageAnalysis,
    FakeOcr,
    FakeQueryEmbedder,
    FakeTextEmbedder,
    InMemoryStorage,
    SyncView,
)
from medlens.adapters.sms import ConsoleSMS
from medlens.ai.gateway.gateway import AsyncLLMGateway
from medlens.ai.gateway.ledger import AsyncLedger
from medlens.core.config import KnowledgeSettings, LLMSettings, Settings
from medlens.core.identifiers import NigerianState
from medlens.core.ids import uuid7
from medlens.core.security import hash_password
from medlens.db.session import UnitOfWork, create_sync_engine, create_sync_session_factory
from medlens.main import create_app
from medlens.modules.audit.models import AuditLog
from medlens.modules.auth.deps import MfaVerifiedUser
from medlens.modules.knowledge.ingestion import ingest_document
from medlens.modules.knowledge.router import get_query_embedder
from medlens.modules.organizations.models import (
    Membership,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.products.models import Product, RegistrationStatus
from medlens.modules.rag.router import get_llm_gateway
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.models import RoleName
from medlens.modules.rbac.permissions import Permission
from medlens.modules.users.models import User
from medlens.modules.verification.pipeline import PipelineConfig, VerificationPipeline
from medlens.modules.verification.scoring import WeightedRuleScorer
from medlens.modules.verification.service import VerificationService
from medlens.ports.barcode import BarcodePort, BarcodeRead
from medlens.ports.image_analysis import ImageAnalysisPort, ImageAssessment
from medlens.ports.llm import LLMPort
from medlens.ports.ocr import OcrEngine, OcrResult
from medlens.resources import Resources, get_storage

PASSWORD = "a-long-unusual-test-passphrase-42"  # pragma: allowlist secret
ROLE_ORG_TYPE: dict[RoleName, OrganizationType] = {
    RoleName.PHARMACY_STAFF: OrganizationType.PHARMACY,
    RoleName.PHARMACY_ADMIN: OrganizationType.PHARMACY,
    RoleName.MANUFACTURER_STAFF: OrganizationType.MANUFACTURER,
    RoleName.MANUFACTURER_ADMIN: OrganizationType.MANUFACTURER,
    RoleName.REGULATOR_ANALYST: OrganizationType.REGULATOR,
    RoleName.REGULATOR_ADMIN: OrganizationType.REGULATOR,
}


def probe_router() -> APIRouter:
    """GET /api/v1/_probe/<permission> for every permission, guarded by require_permissions,
    and GET /api/v1/_probe/mfa, which needs an MFA-verified session."""
    router = APIRouter(prefix="/api/v1/_probe", tags=["admin"])

    @router.get("/mfa")
    async def mfa_probe(principal: MfaVerifiedUser) -> dict[str, str]:
        return {"user_id": str(principal.user_id)}

    for permission in Permission:

        async def probe(permission: Permission = permission) -> dict[str, str]:
            return {"permission": permission.value}

        router.add_api_route(
            f"/{permission.value}",
            probe,
            methods=["GET"],
            dependencies=[Depends(require_permissions(permission))],
            name=f"probe_{permission.name.lower()}",
        )
    return router


def new_phone() -> str:
    return f"+23480{secrets.randbelow(10**8):08d}"


def new_email(label: str = "user") -> str:
    return f"{label}.{uuid4().hex[:10]}@medlens.test"


def new_nafdac_number() -> str:
    return f"Z9-{secrets.randbelow(900000) + 100000}"


def photo(size: tuple[int, int] = (64, 48)) -> bytes:
    """A JPEG of random pixels: every call is a different photo."""
    image = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    output = io.BytesIO()
    image.save(output, "JPEG")
    return output.getvalue()


@dataclass
class Api:
    app: FastAPI
    client: httpx.AsyncClient
    sync_sessions: sessionmaker[Session]
    storage: InMemoryStorage = field(default_factory=InMemoryStorage)
    _counter: int = field(default=0)

    @property
    def resources(self) -> Resources:
        resources: Resources = self.app.state.resources
        return resources

    # ------------------------------------------------------------------ SMS
    def last_code(self, phone: str) -> str:
        sms = self.resources.sms
        assert isinstance(sms, ConsoleSMS)
        for to, message in reversed(sms.sent):
            if to == phone:
                match = re.search(r"\b(\d{6})\b", message)
                assert match
                return match.group(1)
        raise AssertionError(f"no SMS sent to {phone}")

    async def otp_login(self, phone: str) -> dict[str, Any]:
        assert (
            await self.client.post("/api/v1/auth/otp/request", json={"phone": phone})
        ).status_code == 202
        response = await self.client.post(
            "/api/v1/auth/otp/verify", json={"phone": phone, "code": self.last_code(phone)}
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    async def password_login(self, email: str, password: str = PASSWORD) -> httpx.Response:
        return await self.client.post(
            "/api/v1/auth/login", json={"email": email, "password": password}
        )

    # ------------------------------------------------------------------ data
    async def create_user(
        self,
        *,
        email: str | None = None,
        phone: str | None = None,
        password: str | None = PASSWORD,
        global_role: RoleName = RoleName.CONSUMER,
        **fields: Any,
    ) -> User:
        fields.setdefault("password_hash", hash_password(password) if password else None)
        user = User(
            email=email if email is not None else new_email(),
            global_role=global_role.value,
            **fields,
        )
        if phone:
            user.set_phone(phone)
        return await self._save(user)

    async def create_org(
        self,
        kind: OrganizationType,
        *,
        status: OrganizationStatus = OrganizationStatus.ACTIVE,
        jurisdiction: Iterable[str] | None = None,
    ) -> Organization:
        if kind is OrganizationType.REGULATOR and jurisdiction is None:
            jurisdiction = [state.value for state in NigerianState]
        organization = Organization(
            name=f"Test {kind.value} {uuid4().hex[:8]}",
            type=kind,
            status=status,
            jurisdiction_states=list(jurisdiction) if jurisdiction is not None else None,
            verified_at=datetime.now(UTC),
        )
        return await self._save(organization)

    async def add_member(self, user: User, organization: Organization, role: RoleName) -> None:
        await self._save(
            Membership(user_id=user.id, organization_id=organization.id, role=role.value)
        )

    async def user_with_role(self, role: RoleName) -> tuple[User, Organization | None]:
        """A user holding ``role``: global roles directly, organization roles via a fresh org."""
        if role not in ROLE_ORG_TYPE:
            return await self.create_user(global_role=role), None
        organization = await self.create_org(ROLE_ORG_TYPE[role])
        user = await self.create_user()
        await self.add_member(user, organization, role)
        return user, organization

    async def create_product(
        self,
        *,
        brand: str | None = None,
        status: RegistrationStatus = RegistrationStatus.ACTIVE,
        manufacturer: Organization | None = None,
    ) -> Product:
        product = Product(
            nafdac_number=new_nafdac_number(),
            brand_name=brand or f"Testamol {uuid4().hex[:6]}",
            generic_name="Paracetamol",
            strength="500 mg",
            dosage_form="Tablet",
            registration_status=status,
            manufacturer_org_id=manufacturer.id if manufacturer else None,
        )
        return await self._save(product)

    # ------------------------------------------------------------------ worker
    def worker(
        self,
        *,
        text: str = "",
        barcode: str | None = None,
        assessment: ImageAssessment | None = None,
        ocr_confidence: float = 0.9,
        readers: tuple[ImageAnalysisPort, BarcodePort] | None = None,
        ocr: OcrEngine | None = None,
        llm: LLMPort | None = None,
    ) -> VerificationService:
        """The worker's scan processing on this database and storage, with fake readers that
        see ``text`` and ``barcode`` on every photo, or with the given readers."""
        analyzer, barcodes = readers or (
            FakeImageAnalysis(default=assessment),
            FakeBarcode(default=[BarcodeRead("DataMatrix", barcode)] if barcode else []),
        )

        def pipeline(catalog: Any) -> VerificationPipeline:
            return VerificationPipeline(
                catalog=catalog,
                scorer=WeightedRuleScorer(),
                analyzer=analyzer,
                barcode=barcodes,
                ocr=ocr if readers else FakeOcr(default=OcrResult(text, ocr_confidence)),
                llm=llm,
                config=PipelineConfig(llm_enabled=llm is not None),
            )

        return VerificationService(self.sync_sessions, SyncView(self.storage), pipeline)

    # ------------------------------------------------------------------ AI
    def use_llm(self, *script: object, enabled: bool = True) -> AsyncFakeLLMProvider:
        """The assistant's gateway with a scripted provider, metering into this database."""
        provider = AsyncFakeLLMProvider(script=list(script), model="claude-sonnet-5-5")
        gateway = AsyncLLMGateway(
            LLMSettings(enabled=enabled),
            provider,
            AsyncLedger(self.resources.session_factory, self.resources.redis),
        )
        self.app.dependency_overrides[get_llm_gateway] = lambda: gateway
        return provider

    def ingest(self, document_id: object) -> str | None:
        """The worker's ingestion of an uploaded document, with the fake text model."""
        with self.sync_sessions() as session, session.begin():
            status = ingest_document(
                session,
                SyncView(self.storage),
                FakeTextEmbedder(),
                UUID(str(document_id)),
                KnowledgeSettings(),
            )
        return status.value if status else None

    async def reload(self, model: type[Any], key: object) -> Any:
        async with UnitOfWork(self.resources.session_factory) as uow:
            return await uow.session.get(model, key)

    async def audit_actions(self, user_id: UUID | None = None) -> list[str]:
        async with UnitOfWork(self.resources.session_factory) as uow:
            statement = select(AuditLog.action).order_by(AuditLog.created_at, AuditLog.id)
            if user_id is not None:
                statement = statement.where(AuditLog.actor_user_id == user_id)
            return list(await uow.session.scalars(statement))

    async def _save(self, entity: Any) -> Any:
        async with UnitOfWork(self.resources.session_factory) as uow:
            uow.session.add(entity)
            await uow.commit()
        return entity

    # ------------------------------------------------------------------ tokens
    def token(
        self,
        user: User,
        *,
        organization: Organization | None = None,
        auth_methods: tuple[str, ...] = ("pwd",),
        perm_ver: int | None = None,
        session_id: UUID | None = None,
        issued_at: int | None = None,
    ) -> str:
        return self.resources.keyring.issue(
            "access",
            subject=user.id,
            session_id=session_id or uuid7(),
            organization_id=organization.id if organization else None,
            perm_ver=user.perm_ver if perm_ver is None else perm_ver,
            auth_methods=auth_methods,
            now=issued_at,
        )

    @staticmethod
    def bearer(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}


@asynccontextmanager
async def open_api(settings: Settings) -> AsyncIterator[Api]:
    """The app (plus the permission probe routes) running on ``settings``, Redis flushed."""
    app = create_app(settings)
    app.include_router(probe_router())
    storage = InMemoryStorage()
    app.dependency_overrides[get_storage] = lambda: storage
    query_embedder = FakeQueryEmbedder(FakeTextEmbedder())
    app.dependency_overrides[get_query_embedder] = lambda: query_embedder
    engine = create_sync_engine(settings.database, application_name="medlens-tests")
    try:
        async with app.router.lifespan_context(app):
            await app.state.resources.redis.flushdb()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                yield Api(app, client, create_sync_session_factory(engine), storage)
    finally:
        engine.dispose()
