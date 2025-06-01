import torch
import math


class AngleVectorizer:
    """
    Encodes/decodes angular directions (azimuth, elevation) as 3D unit vectors.
    This representation is beneficial for neural networks as it handles the wrapping
    nature of angles (e.g., 0° is close to 359°) seamlessly.

    The coordinate system used is:
    - Azimuth (az): angle in the XY plane, measured from the positive X-axis
                    towards the positive Y-axis.
                    Expected input range: 0 to 360 degrees (or 0 to 2π radians).
                    Output range: 0 to 360 degrees (or 0 to 2π radians).
    - Elevation (el): angle from the XY plane towards the Z-axis.
                      Positive elevation goes towards positive Z.
                      Expected input range: -90 to 90 degrees (or -π/2 to π/2 radians).
                      Output range: -90 to 90 degrees (or -π/2 to π/2 radians).
    - Vector components are derived as:
        x = cos(el) * cos(az)
        y = cos(el) * sin(az)
        z = sin(el)
    """

    def __init__(self, angle_units='degrees'):
        """
        Initializes the AngleVectorizer.

        Args:
            angle_units (str, optional): The units for input and output angles.
                                         Must be either 'degrees' or 'radians'.
                                         Defaults to 'degrees'.
        """
        if angle_units not in ['degrees', 'radians']:
            raise ValueError("angle_units must be 'degrees' or 'radians'")
        self.angle_units = angle_units
        # Use torch.pi if available (PyTorch 1.7+), otherwise math.pi
        self.pi = getattr(torch, 'pi', math.pi)

    def _process_input_angles(self, az, el):
        """
        Converts azimuth and elevation inputs to float tensors, ensures they are on
        the same device, and converts them to radians if necessary.
        """
        # Convert 'az' to a float tensor
        if not isinstance(az, torch.Tensor):
            az_tensor = torch.tensor(az, dtype=torch.float32)
        else:
            az_tensor = az.float()

        # Convert 'el' to a float tensor, matching 'az_tensor's device
        if not isinstance(el, torch.Tensor):
            el_tensor = torch.tensor(el, dtype=torch.float32, device=az_tensor.device)
        else:
            el_tensor = el.float()

        # Ensure 'el_tensor' is on the same device as 'az_tensor'
        if el_tensor.device != az_tensor.device:
            el_tensor = el_tensor.to(az_tensor.device)

        # Broadcast shapes if necessary (e.g. scalar az with batched el)
        # This makes them have the same shape for element-wise operations
        try:
            az_tensor, el_tensor = torch.broadcast_tensors(az_tensor, el_tensor)
        except RuntimeError as e:
            raise ValueError(
                f"Azimuth and elevation shapes are not broadcastable: az_shape={az_tensor.shape}, el_shape={el_tensor.shape}. Error: {e}")

        if self.angle_units == 'degrees':
            az_rad = torch.deg2rad(az_tensor)
            el_rad = torch.deg2rad(el_tensor)
        else:  # radians
            az_rad = az_tensor
            el_rad = el_tensor

        return az_rad, el_rad

    def angles_to_vector(self, az, el):
        """
        Converts azimuth and elevation angle(s) to a 3D unit vector.

        Args:
            az (torch.Tensor or float or list): Azimuth angle(s).
            el (torch.Tensor or float or list): Elevation angle(s).
               Angles are assumed to be in the units specified during class
               initialization ('degrees' or 'radians').
               Azimuth is typically in [0, 360] or [0, 2π].
               Elevation is typically in [-90, 90] or [-π/2, π/2].

        Returns:
            torch.Tensor: 3D unit vector(s). If inputs are scalars, shape is [3].
                          If inputs are 1D tensors of length B, shape is [B, 3].
                          In general, if inputs broadcast to shape S, output is S + [3].
        """
        az_rad, el_rad = self._process_input_angles(az, el)

        # Spherical to Cartesian conversion formulas
        x = torch.cos(el_rad) * torch.cos(az_rad)
        y = torch.cos(el_rad) * torch.sin(az_rad)
        z = torch.sin(el_rad)

        # Stack along a new last dimension.
        # Example: if az_rad, el_rad are shape [B], output is [B, 3].
        # Example: if az_rad, el_rad are scalars, output is [3].
        target_vec = torch.stack([x, y, z], dim=-1)
        return target_vec

    def vector_to_angles(self, vec):
        """
        Converts 3D unit vector(s) back to azimuth and elevation angles.

        Args:
            vec (torch.Tensor or list/tuple of lists): 3D unit vector(s).
                 Expected shape [..., 3] (e.g., a single [x,y,z] list or
                 a batch tensor of shape [B, 3]).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - az_out: Azimuth angle(s) in the specified units.
                          Range [0, 360) degrees or [0, 2π) radians.
                - el_out: Elevation angle(s) in the specified units.
                          Range [-90, 90] degrees or [-π/2, π/2] radians.
        """
        if not isinstance(vec, torch.Tensor):
            # Attempt to convert to tensor, assuming input might be e.g. list of lists
            try:
                vec_tensor = torch.tensor(vec, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Input 'vec' could not be converted to a tensor. Original error: {e}")
        else:
            vec_tensor = vec.float()

        if vec_tensor.shape[-1] != 3:
            raise ValueError(
                f"Input vector must have its last dimension of size 3 (for x,y,z), "
                f"but got shape {vec_tensor.shape}"
            )

        x = vec_tensor[..., 0]
        y = vec_tensor[..., 1]
        z = vec_tensor[..., 2]

        # Clamp z to avoid domain errors with asin for numerically unstable vectors.
        # This can happen if network predictions are not perfectly unit vectors (norm slightly > 1).
        z = torch.clamp(z, -1.0, 1.0)

        # Cartesian to Spherical conversion
        # Elevation: range –π/2 to π/2 radians
        el_rad = torch.asin(z)
        # Azimuth: range –π to π radians (output of atan2)
        az_rad = torch.atan2(y, x)

        # Normalize azimuth to be in [0, 2π) range for radians,
        # or [0, 360) for degrees, matching the prompt's example.
        # torch.remainder with a positive divisor maps the input into [0, divisor).
        # e.g., torch.remainder(-pi/2, 2*pi) = 3*pi/2.
        az_rad_normalized = torch.remainder(az_rad, 2 * self.pi)

        if self.angle_units == 'degrees':
            az_out = torch.rad2deg(az_rad_normalized)
            el_out = torch.rad2deg(el_rad)
        else:  # radians
            az_out = az_rad_normalized
            el_out = el_rad  # el_rad is already in –π/2 to π/2

        return az_out, el_out


# --- Example Usage ---
if __name__ == '__main__':
    # --- Test with degrees ---
    print("--- Testing with Degrees ---")
    vectorizer_deg = AngleVectorizer(angle_units='degrees')

    # Single angle pair
    az_deg_single, el_deg_single = 45.0, 30.0
    vec_single = vectorizer_deg.angles_to_vector(az_deg_single, el_deg_single)
    print(f"Input (deg): az={az_deg_single}, el={el_deg_single}")
    print(f"Vector: {vec_single.numpy()}")
    az_rec_single, el_rec_single = vectorizer_deg.vector_to_angles(vec_single)
    print(f"Recovered (deg): az={az_rec_single.item():.2f}, el={el_rec_single.item():.2f}\n")

    # Batch of angles (as lists)
    az_deg_batch_list = [0.0, 90.0, 180.0, 359.0]
    el_deg_batch_list = [0.0, 45.0, -30.0, 0.0]
    vec_batch_list = vectorizer_deg.angles_to_vector(az_deg_batch_list, el_deg_batch_list)
    print(f"Input Batch (deg, lists): az={az_deg_batch_list}, el={el_deg_batch_list}")
    print(f"Vector Batch:\n{vec_batch_list.numpy()}")
    az_rec_batch_list, el_rec_batch_list = vectorizer_deg.vector_to_angles(vec_batch_list)
    print(f"Recovered Batch (deg):")
    for i in range(len(az_deg_batch_list)):
        print(f"  az={az_rec_batch_list[i].item():.2f}, el={el_rec_batch_list[i].item():.2f}")
    print("")

    # Batch of angles (as PyTorch tensors)
    az_deg_batch_tensor = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float32)
    el_deg_batch_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    vec_batch_tensor = vectorizer_deg.angles_to_vector(az_deg_batch_tensor, el_deg_batch_tensor)
    print(f"Input Batch (deg, tensors): az={az_deg_batch_tensor}, el={el_deg_batch_tensor}")
    print(f"Vector Batch:\n{vec_batch_tensor.numpy()}")
    az_rec_batch_tensor, el_rec_batch_tensor = vectorizer_deg.vector_to_angles(vec_batch_tensor)
    print(f"Recovered Batch (deg):")
    for i in range(az_deg_batch_tensor.shape[0]):
        print(f"  az={az_rec_batch_tensor[i].item():.2f}, el={el_rec_batch_tensor[i].item():.2f}")
    print("")

    # Test wrapping for azimuth 0 and 359 degrees (should be close vectors)
    az1, el1 = 0.0, 0.0
    az2, el2 = 359.0, 0.0
    vec1 = vectorizer_deg.angles_to_vector(az1, el1)
    vec2 = vectorizer_deg.angles_to_vector(az2, el2)
    print(f"Vector for az=0, el=0: {vec1.numpy()}")  # Should be close to [1,0,0]
    print(f"Vector for az=359, el=0: {vec2.numpy()}")  # Should be close to [1,0,0]
    dist = torch.norm(vec1 - vec2)
    print(f"Distance between vectors for 0 deg and 359 deg: {dist.item():.4f}\n")

    # --- Test with radians ---
    print("--- Testing with Radians ---")
    vectorizer_rad = AngleVectorizer(angle_units='radians')

    # Single angle pair
    az_rad_single, el_rad_single = math.pi / 4, math.pi / 6  # 45 deg, 30 deg
    vec_rad_single = vectorizer_rad.angles_to_vector(az_rad_single, el_rad_single)
    print(f"Input (rad): az={az_rad_single:.2f}, el={el_rad_single:.2f}")
    print(f"Vector: {vec_rad_single.numpy()}")
    az_rec_rad_single, el_rec_rad_single = vectorizer_rad.vector_to_angles(vec_rad_single)
    print(f"Recovered (rad): az={az_rec_rad_single.item():.2f}, el={el_rec_rad_single.item():.2f}\n")

    # Test wrapping for azimuth 0 and nearly 2*pi radians
    az_r1, el_r1 = 0.0, 0.0
    az_r2, el_r2 = (2 * math.pi) - 0.017, 0.0  # Approx 359 deg
    vec_r1 = vectorizer_rad.angles_to_vector(az_r1, el_r1)
    vec_r2 = vectorizer_rad.angles_to_vector(az_r2, el_r2)
    print(f"Vector for az=0 rad, el=0 rad: {vec_r1.numpy()}")
    print(f"Vector for az~2pi rad, el=0 rad: {vec_r2.numpy()}")
    dist_rad = torch.norm(vec_r1 - vec_r2)
    print(f"Distance between vectors for 0 rad and ~2pi rad: {dist_rad.item():.4f}\n")

    # Test pole (elevation = 90 degrees or pi/2 radians)
    # Azimuth should be 0 by convention from atan2(0,0)
    az_pole_deg, el_pole_deg = 123.0, 90.0
    vec_pole = vectorizer_deg.angles_to_vector(az_pole_deg, el_pole_deg)
    print(f"Input (deg): az={az_pole_deg}, el={el_pole_deg}")
    print(f"Vector at pole: {vec_pole.numpy()}")  # Should be [0,0,1]
    az_rec_pole, el_rec_pole = vectorizer_deg.vector_to_angles(vec_pole)
    print(f"Recovered from pole (deg): az={az_rec_pole.item():.2f}, el={el_rec_pole.item():.2f}\n")

    # Test broadcasting
    print("--- Testing Broadcasting ---")
    az_scalar_deg = 45.0
    el_batch_deg = torch.tensor([0.0, 30.0, 60.0])
    vec_broadcast = vectorizer_deg.angles_to_vector(az_scalar_deg, el_batch_deg)
    print(f"Input (deg): az_scalar={az_scalar_deg}, el_batch={el_batch_deg.numpy()}")
    print(f"Vector (broadcasted):\n{vec_broadcast.numpy()}")
    az_rec_broadcast, el_rec_broadcast = vectorizer_deg.vector_to_angles(vec_broadcast)
    print(f"Recovered Batch (deg):")
    for i in range(el_batch_deg.shape[0]):
        print(f"  az={az_rec_broadcast[i].item():.2f}, el={el_rec_broadcast[i].item():.2f}")