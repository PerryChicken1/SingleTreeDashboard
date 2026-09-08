"""Per-run WLS environments; credentials never enter shared solver objects."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from uuid import UUID

_environment = ContextVar("singletree_gurobi_environment", default=None)
_initialization_lock = Lock()


class GurobiLicenseError(RuntimeError):
    """A user-facing error that contains no credential values."""


def wls_credentials(access_id: str, secret: str, license_id: str) -> dict:
    """Validate the three WLS fields without echoing their contents."""
    access_id, secret, license_id = access_id.strip(), secret.strip(), license_id.strip()
    if not all((access_id, secret, license_id)):
        raise GurobiLicenseError("Enter your Gurobi WLS access ID, secret key and licence ID.")
    try:
        UUID(access_id)
        UUID(secret)
    except ValueError:
        raise GurobiLicenseError("The WLS access ID and secret key must be valid UUIDs from your Gurobi WLS API key.") from None
    if len(license_id) > 10 or not license_id.isascii() or not license_id.isdecimal() or not 0 < int(license_id) <= 2147483647:
        raise GurobiLicenseError("The Gurobi licence ID must be a positive integer.")
    return {"WLSACCESSID": access_id, "WLSSECRET": secret, "LICENSEID": int(license_id)}


def current_gurobi_environment():
    """Return only this execution context's environment, never another user's."""
    return _environment.get()


@contextmanager
def gurobi_wls_session(credentials: dict):
    """Authenticate once per operation and release resources on every exit."""
    try:
        import gurobipy as gp
    except ImportError:
        raise GurobiLicenseError("Gurobi is not installed on this server. Use CBC or install the dashboard requirements.") from None

    env = None
    token = None
    try:
        # Gurobi environment initialization is not thread-safe. Solves use
        # separate environments and can proceed concurrently after this block.
        with _initialization_lock:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("LogToConsole", 0)
            env.setParam("LogFile", "")
            for name in ("WLSACCESSID", "WLSSECRET", "LICENSEID"):
                env.setParam(name, credentials[name])
            env.start()
        token = _environment.set(env)
        yield
    except gp.GurobiError:
        raise GurobiLicenseError(
            "Gurobi could not complete the request. Check your WLS credentials, "
            "licence validity, available sessions and permission to run on this cloud host. "
            "You can also select CBC."
        ) from None
    finally:
        if token is not None:
            _environment.reset(token)
        if env is not None:
            env.dispose()
