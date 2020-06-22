def update_halo(dim, field, num_halo ):
    """Update the halo-zone using an up/down and left/right strategy.
    
    field    -- input/output field (nz x ny x nx with halo in x- and y-direction)
    num_halo -- number of halo points
    
    Note: corners are updated in the left/right phase of the halo-update
    """
    if dim == 3:
        # bottom edge (without corners)
        field[:, 0:num_halo, num_halo:-num_halo] = field[:, -2 * num_halo:-num_halo, num_halo:-num_halo]

        # top edge (without corners)
        field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo:2 * num_halo, num_halo:-num_halo]

        # left edge (including corners)
        field[:, :, 0:num_halo] = field[:, :, -2 * num_halo:-num_halo]

        # right edge (including corners)
        field[:, :, -num_halo:] = field[:, :, num_halo:2 * num_halo]
    if dim == 2:
        # bottom edge
        field[:,0:num_halo] = field[:, -2*num_halo:-num_halo]
        # top edge
        field[:, -num_halo:] = field[:, num_halo:2*num_halo]
        # left edge
        field[0:num_halo, :] = field[-2*num_halo:-num_halo, :]
        # right edge
        field[-num_halo:, :] = field[num_halo:2*num_halo, :]
        
    if dim == 1:
        # left edge
        field[0:num_halo] = field[-2*num_halo:-num_halo]
        # right edge
        field[-num_halo:] = field[num_halo:2*num_halo]
    return field