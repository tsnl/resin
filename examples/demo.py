from zero import lib as zero


def main():
    engine = zero.init(
        zero.config(
            True,
            True,
            True,
        )
    )
    zero.run(engine)


if __name__ == "__main__":
    main()
