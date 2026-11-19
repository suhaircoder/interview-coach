import os
from urllib.request import getproxies

from dotenv import load_dotenv


def configure_network() -> None:
    """Prevent HTTPX from loading unsupported SOCKS4 system proxies."""
    proxy_urls = (str(value).lower() for value in getproxies().values())
    if any(url.startswith("socks4://") for url in proxy_urls):
        os.environ["NO_PROXY"] = "*"


def main() -> None:
    load_dotenv()
    configure_network()

    # Import after loading .env so libraries can read network configuration
    # such as NO_PROXY during initialization.
    from src.ui import create_app

    create_app().launch()


if __name__ == "__main__":
    main()
