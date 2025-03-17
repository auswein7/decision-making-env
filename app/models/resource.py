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
