def ori_position(A_pos, A_orient, B_pos):
  orientation_transform = {
    0: lambda x, y: (x, y),
    1: lambda x, y: (y, -x),
    2: lambda x, y: (-x, -y),
    3: lambda x, y: (-y, x),
  }
  dx = B_pos[0] - A_pos[0]
  dy = B_pos[1] - A_pos[1]
  return orientation_transform[int(A_orient)](dx, dy)