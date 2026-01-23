import resin.cc as cc


@cc.function
def fibonacci(n: cc.Int) -> cc.Int:
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
