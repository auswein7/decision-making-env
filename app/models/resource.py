class Resource:
    """
    Simple class representing a resource.

    Attributes:
        id: integer value representing the resource id.
        value: integer value representing the resource value.
    """

    def __init__(self, resource_id, value):
        self.id = resource_id
        self.value = value

    #TODO::USE THIS FUNCTION THROUGHOUT THE CODE
    def __repr__(self):
        return "Resource(id={}, value={})".format(self.id, self.value)