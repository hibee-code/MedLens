# Roles and permissions

MedLens uses role-based access control at two levels:

1. **Permissions** (`resource:action`) decide which endpoints a principal may call. A user's
   permissions are the union of their **global role** (`users.global_role`) and the role of
   their membership in the **active organization** (the `org` claim of the access token, chosen
   with `POST /api/v1/auth/switch-organization`). Memberships in suspended, unverified or
   deleted organizations grant nothing.
2. **Policies** (`src/medlens/modules/rbac/policies.py`) decide which *rows* a permission
   reaches. They are called from services, never routers, so workers and the API share them.

The matrix below is the baseline seeded by migration `0008`. At runtime permissions are read
from the `roles` and `role_permissions` tables (cached in Redis), so the matrix can change
without a deploy. A test fails if this table, the code (`permissions.py`) and a freshly
migrated database disagree.

## Role → permission matrix

| Permission | `consumer` | `health_worker` | `pharmacy_staff` | `pharmacy_admin` | `manufacturer_staff` | `manufacturer_admin` | `regulator_analyst` | `regulator_admin` | `platform_admin` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `scan:create` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `scan:read_own` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `scan:read_org` |  |  | ✓ | ✓ |  |  |  |  | ✓ |
| `scan:read_all` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `scan:read_signals` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `product:read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `product:write` |  |  |  |  | ✓ | ✓ |  |  | ✓ |
| `product:review` |  |  |  |  |  |  |  | ✓ | ✓ |
| `batch:write` |  |  |  |  | ✓ | ✓ |  |  | ✓ |
| `reference_image:write` |  |  |  |  | ✓ | ✓ |  |  | ✓ |
| `case:read` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `case:review` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `case:assign` |  |  |  |  |  |  |  | ✓ | ✓ |
| `hotspot:read` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `hotspot:export` |  |  |  |  |  |  |  | ✓ | ✓ |
| `org:read` |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `org:manage` |  |  |  | ✓ |  | ✓ |  | ✓ | ✓ |
| `member:invite` |  |  |  | ✓ |  | ✓ |  | ✓ | ✓ |
| `member:manage` |  |  |  | ✓ |  | ✓ |  | ✓ | ✓ |
| `apikey:manage` |  |  |  | ✓ |  | ✓ |  | ✓ | ✓ |
| `audit:read` |  |  |  |  |  |  |  |  | ✓ |
| `reports:read_own` |  |  |  |  | ✓ | ✓ |  |  | ✓ |
| `knowledge:read` |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `knowledge:manage` |  |  |  |  |  |  | ✓ | ✓ | ✓ |
| `user:manage` |  |  |  |  |  |  |  |  | ✓ |
| `platform:admin` |  |  |  |  |  |  |  |  | ✓ |

Global roles: `consumer` (default for every user), `health_worker`, `platform_admin`.
Organization roles and the organization types that can hold them:

| Role | Organization types |
|---|---|
| `pharmacy_staff`, `pharmacy_admin` | pharmacy, ppmv, clinic, distributor |
| `manufacturer_staff`, `manufacturer_admin` | manufacturer |
| `regulator_analyst`, `regulator_admin` | regulator |

There is no distributor role: distributors use the pharmacy roles until bulk stock verification
is built, which is when their needs start to differ.

## Policies (row-level rules)

| Data | Who sees or changes what |
|---|---|
| Scans | Consumers and health workers: their own. Pharmacy (and distributor) staff: scans made under their organization. Regulators: scans in their jurisdiction (`jurisdiction_states`; national regulators list all 37). Platform admins: all. Manufacturers never see raw scans, not even of their own products (scans carry locations and reveal who checks what); they get aggregates (see the roadmap). |
| Scan signals | Only `scan:read_signals` (regulators, platform admins) sees the detailed signals behind a verdict, so counterfeiters cannot learn which checks they fail. Everyone else gets reason codes. |
| Cases, hotspots | Regulators: inside their jurisdiction. Platform admins: all. |
| Knowledge documents | Every `knowledge:read` holder reads documents marked `professional`; documents marked `regulator` are for regulators and platform admins only. Consumers never reach the knowledge base or the assistant. |
| Reports | Manufacturers (`reports:read_own`): flagged-scan counts for their own products by state and ISO week, cells under 5 suppressed. Regulators (`scan:read_all`): the scan summary within their jurisdiction. |
| Products, batches, reference images | Manufacturers write only products linked to their organization. Platform admins: all. |
| Organizations, members | Organization admins act only on their own organization; other organizations look like they do not exist (404, not 403). Platform admins act on any. |
| Audit log | Platform admins only (`audit:read`). The table is append-only: a trigger rejects `UPDATE`, `DELETE` and `TRUNCATE` for every role (migration `0005`), and the application role has no such privileges on it either (migration `0006`). |

## Changing access safely

* **No escalation.** Nobody can grant a role that carries a permission they do not hold
  themselves, and organization roles must suit the organization's type.
* **Last admin.** An organization always keeps at least one active admin
  (`pharmacy_admin`, `manufacturer_admin` or `regulator_admin`): the last one can be neither
  removed nor demoted, and cannot erase their account.
* **Immediate effect.** Every role or membership change increments the user's `perm_ver` (in
  the database and in Redis). Permissions are cached per `perm_ver`, so the next request uses
  the new permissions even with an old access token. Sensitive endpoints (changing roles,
  managing API keys) go further and reject tokens issued before the change, forcing a refresh.
* **Audit.** Every change is written to `audit_logs`.

## Roadmap

Planned permissions, not in the matrix yet (`reports:read_own`, the manufacturer report, has
shipped: `GET /api/v1/manufacturer/scans`):

| Permission | For | Scope |
|---|---|---|
| `audit:read_regulatory` | `regulator_analyst`, `regulator_admin` | Audit entries about recalls, approvals and product-status changes only, within the regulator's jurisdiction; never authentication or account entries. |

## API keys

Organization API keys (`ml_live_<prefix>_<secret>`) carry explicit scopes, chosen when the key
is created. Scopes must be permissions the creator holds, and never administrative ones
(`org:manage`, `member:*`, `apikey:manage`, `audit:read`, `user:manage`, `platform:admin`), nor
approvals (`product:review`): consequential decisions are made by people (CLAUDE.md §5).
