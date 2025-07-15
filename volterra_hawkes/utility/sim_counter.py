class SimCounter:
    counter: int

    def __init__(self):
        self.counter = 0

    def add(self, x: int):
        self.counter += x

    def __str__(self):
        return str(self.counter)

    def __repr__(self):
        print("Counter: ", self.counter)
