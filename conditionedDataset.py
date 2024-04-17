import os
import torch
from torch.utils.data import Dataset

class AirfoilDataset(Dataset):
    def __init__(self, path='cleaned_airfoils/', eval_path='evaluated_airfoils/'):
        super(AirfoilDataset, self).__init__()
        self._X = torch.tensor([])  # Initialize as empty tensor
        self._Y = []  # List to hold tensors for each airfoil
        self.names = []  # Name of all airfoils
        self.norm_coeff = 0  # Normalization coeff to scale y to [-1, 1]
        self.ClCd = []  # List to hold Cl and Cd coefficients

        airfoil_fn = [afn for afn in os.listdir(path) if afn.endswith('.dat')]
        all_y = []  # Temporary list to hold all y values for normalization

        # Process each airfoil file for geometry
        for idx, fn in enumerate(airfoil_fn):
            with open(os.path.join(path, fn), 'r', encoding="utf8", errors='ignore') as f:
                self.names.append(fn[:-4])  # Assuming .dat extension
                raw_data = f.readlines()
                x_coords, y_coords = [], []
                for line in raw_data:
                    parts = line.split()
                    if len(parts) == 2:
                        x, y = map(float, parts)
                        x_coords.append(x)
                        y_coords.append(y)
                if idx == 0:  # Only need to set _X once since x coordinates are shared
                    self._X = torch.tensor(x_coords, dtype=torch.float32)
                y_tensor = torch.tensor(y_coords, dtype=torch.float32)
                all_y.append(y_tensor)
                self._Y.append(y_tensor)

        # Process each airfoil file for Cl/Cd
        for name in self.names:
            file_processed = False  # Flag to indicate if the file's valid line has been processed
            with open(os.path.join(eval_path, name), 'r', encoding="utf8", errors='ignore') as f:
                for line in f:
                    if 'alpha' in line:
                        file_processed = True  # Start processing data lines after this header line
                    elif file_processed:
                        parts = line.split()
                        if len(parts) == 9:
                            try:
                                # Attempt to parse the Cl and Cd values
                                Cl, Cd = float(parts[1]), float(parts[2])
                                self.ClCd.append((Cl, Cd))
                                break  # Break after processing the first valid data line
                            except ValueError:
                                continue  # Skip lines where conversion to float fails

        # Determine normalization coefficient from all y values
        all_y_tensor = torch.cat(all_y)
        self.norm_coeff = torch.max(torch.abs(all_y_tensor))

        # Normalize each y tensor
        self._Y = [(y / self.norm_coeff) for y in self._Y]
        self.ClCd = torch.tensor(self.ClCd, dtype=torch.float32)  # Convert list of tuples to tensor

    def __len__(self):
        return len(self._Y)

    def __getitem__(self, idx):
        # Include Cl and Cd coefficients along with y coordinates
        ClCd = self.ClCd[idx]
        print(f"CL/CD values at index {idx}: {ClCd}")
        return self._Y[idx].clone().unsqueeze(0), self.ClCd[idx]

    def get_x(self):
        return self._X.clone()

    def get_y(self):
        return [y.clone() for y in self._Y]

    def get_clcd(self):
        return self.ClCd.clone()


