
def get_rigid(group_element):
    return group_element[1]

def get_scale(group_element):
    return group_element[0]

def create_signed_element(sign, group_element):
    return (sign, group_element)

def get_sign(signed_group_element):
    return signed_group_element[0]

def get_group_element(signed_group_element):
    return signed_group_element[1]
