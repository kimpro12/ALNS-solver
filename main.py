"""Entry point delegating to the chapter 10 glue pipeline CLI."""

from alns_cvrptwpd.glue.pipeline import main as pipeline_main


def main() -> None:
    pipeline_main()


if __name__ == "__main__":
    main()
