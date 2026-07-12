class History:

    def __init__(self):
        self.records = {}

    def add(self, key, value):

        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value)

    def get(self, key):
        return self.records.get(key, [])

    def keys(self):
        return list(self.records.keys())

    def __getitem__(self, key):
        return self.records[key]

    def __repr__(self):
        return f"History({self.records})"
