"""python -m medlens.seed [--data-dir DIR]"""

import argparse
import sys
from pathlib import Path

from medlens.core.bootstrap import configure_process
from medlens.core.config import Environment, get_settings
from medlens.db.session import create_sync_engine, create_sync_session_factory, sync_session_scope
from medlens.modules.auth.passwords import password_problems
from medlens.seed import DEFAULT_DATA_DIR, run_seed
from medlens.seed.greenbook import clear_cached_products


def main() -> int:
    parser = argparse.ArgumentParser(description="Load development seed data (idempotent).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--staff-password",
        help="password for the e-mail test users (set only where none exists yet)",
    )
    parser.add_argument(
        "--reset-staff-logins",
        action="store_true",
        help="with --staff-password: replace the e-mail test users' passwords and turn their MFA "
        "off (local demos)",
    )
    args = parser.parse_args()
    if args.reset_staff_logins and not args.staff_password:
        sys.stderr.write("--reset-staff-logins needs --staff-password\n")
        return 2
    if args.staff_password and (problems := password_problems(args.staff_password)):
        sys.stderr.write("--staff-password " + "; ".join(problems) + "\n")
        return 2

    settings = get_settings()
    if settings.app.environment in {Environment.STAGING, Environment.PRODUCTION}:
        sys.stderr.write(f"refusing to seed a {settings.app.environment} database\n")
        return 1
    configure_process(settings)

    engine = create_sync_engine(settings.database, application_name="medlens-seed")
    try:
        with sync_session_scope(create_sync_session_factory(engine)) as session:
            summary = run_seed(
                session,
                args.data_dir,
                staff_password=args.staff_password,
                reset_staff_logins=args.reset_staff_logins,
            )
    finally:
        engine.dispose()
    clear_cached_products(settings)
    sys.stdout.write(
        f"seeded {summary.products} products, {summary.organizations} organizations, "
        f"{summary.batches} batches, {summary.users} users, {summary.memberships} memberships, "
        f"{summary.documents} knowledge documents\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
