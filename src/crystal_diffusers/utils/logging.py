import sys
from loguru import logger

logger.disable("crystal_diffusers")


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str | None = None,
):
    """
    Configure Loguru with two handlers:
      1) DEBUG & INFO -> stdout  (business output)
      2) WARNING & above -> stderr (diagnostics)
    """
    logger.enable("crystal_diffusers")

    # clear any existing handlers
    logger.remove()

    default_format = (
        "<cyan>{time:YY-MM-DD HH:mm:ss}</cyan> │ "
        "<level>{level:^8}</level> │ "
        "<green>{name}</green>:<green>{function}</green>:<green>{line}</green> — "
        "{message}"
    )

    fmt = fmt or default_format

    # numeric value of WARNING so we can split streams
    warn_no = logger.level("WARNING").no

    logger.add(
        sys.stdout,
        level=level,
        format=fmt,
        colorize=True,
        filter=lambda rec: rec["level"].no < warn_no,
    )

    user_no = logger.level(level).no
    stderr_level = level if user_no > warn_no else "WARNING"

    logger.add(
        sys.stderr,
        level=stderr_level,
        format=fmt,
        colorize=True,
    )
