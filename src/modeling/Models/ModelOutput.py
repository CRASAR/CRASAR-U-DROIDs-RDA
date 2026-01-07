class ModelOutput:
    def __init__(self):
        self.__fields = {}
    def __getitem__(self, field):
        return self.__fields[field]
    def setField(self, field, value):
        self.__fields[field] = value
    def contains(self, field):
        return field in self.__fields
