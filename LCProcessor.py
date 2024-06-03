# Standard library imports
import math
import time

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import multi_dot
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

class LCProcessor:
  def __init__(self, n_para, n_perp, ks=None):
    '''
    Initialize the MyDataFrameProcessor class with parameters for processing
        liquid crystal samples.
    Inputs:
      n_para: Extraordinary index of refraction, corresponding to light
          polarization parallel to the optical axis.
      n_perp: Ordinary index of refraction, corresponding to light polarization
          perpendicular to the optical axis.
      (Optional) ks: list of Wavenumbers (2*pi/lambda), where lambda is the
          wavelength of light.
    '''
    if ks == None:
      self.Nk = 5
      self.ks = np.linspace(2*math.pi/700/10E-9,
                            2*math.pi/400/10E-9, num=self.Nk)
    else:
      self.ks = ks
    self.n_para = n_para
    self.n_perp = n_perp

  # 3D director fields
  def make_mesh_3D(self,Lx,Ly,Lz,a):
    #L width of the square box, a resolution
    x_lin=np.linspace(-Lx/2,+Lx/2,int(Lx/a))
    y_lin=np.linspace(-Ly/2,+Ly/2,int(Ly/a))
    z_lin=np.linspace(-Lz/2,+Lz/2,int(Lz/a))
    X,Y,Z=np.meshgrid(x_lin,y_lin,z_lin,indexing='ij')
    return X,Y,Z

  def compute_phi_defect_2D(self,s,xc,yc,phi_0,X,Y):
    #s winding number
    #xc, yc coordinates of the defect
    #phi_0 offset (global rotation of the director field)
    #X,Y coordinates where the field phi is evaluated, can be arrays
    phi=s*np.arctan2(Y-yc,X-xc)+phi_0
    return phi

  def parse_dataframe_from_list(self,X,Y,Z,isLC,Nx=[],Ny=[],Nz=[],theta=[],phi=[]):
    return

  def create_defect_dataframe(self, Nz, d, spin=1/2):
    '''
    Constructs a defect dataframe. The z coordinates will range
    from -Nz*dz/2 to Nz*dz/2. X and Y range from -50*dz to 50*dz
    Inputs:
      Nz: number of Z coordinates
      d: resolution
      (optional) spin: defect spin
    '''
    data = []
    self.dz = d

    (x,y,z) = self.make_mesh_3D(100*d,100*d,Nz*d,d)

    data = {
        'X': x.ravel(),
        'Y': y.ravel(),
        'Z': z.ravel(),
        'theta': np.full(100*100*Nz, np.pi / 2),
        'phi': self.compute_phi_defect_2D(spin, 0, 0, 0, x, y).flatten()
    }

    # create a data frame to store the input file
    columns = ['X', 'Y', 'Z', 'theta', 'phi']
    self.df = pd.DataFrame(data, columns=columns)
    self.df = self.df.sort_values(by=['X', 'Y', 'Z'])

    self.establish_mesh()

  def parse_dataframe_from_txt(self, file_path):
    '''
    Reads a text file and constructs a DataFrame from its contents. This method
        is intended for processing files from COMSOL.
    Inputs:
      file_path: The path to the text file containing data to be processed.
          The expected format involves columns representing spatial coordinates
          and vector components, among other possible data.
    '''
    data = []
    with open(file_path, 'r') as file:
      for line in file:
        if not line.startswith('%'):
          # Collect data
          if line.strip():
            values = line.split()
            if len(values) == 6:
              # Compute theta and phi from the director vector
              director_XYZ = [float(val) for val in values[3:]]
              theta = math.atan2(math.sqrt(director_XYZ[1] * director_XYZ[1] + director_XYZ[0] * director_XYZ[0]), director_XYZ[2])
              phi = math.atan2(director_XYZ[1], director_XYZ[0])
              # Data stored with columns  X, Y, Z, theta, phi
              data.append((float(values[0]), float(values[1]), float(values[2]), theta, phi))

      # create a data frame to store the input file
      columns = ['X', 'Y', 'Z', 'theta', 'phi']
      self.df = pd.DataFrame(data, columns=columns)
      self.df = self.df.sort_values(by=['X', 'Y', 'Z'])

      self.establish_mesh()

  def establish_mesh(self):
    '''
    Establishes a spatial mesh grid based on the x, y, and z coordinates present
        in the DataFrame. This mesh is used for subsequent spatial analyses and
        visualizations. It is automatically perfomed when
        parse_dataframe_from_txt is called.
    Note: This method relies on the DataFrame being properly populated, either
        through initialization or by parsing a text file.
    '''
    #list the x-coordinates of the data points, in increasing order
    XL=np.unique(self.df['X'])
    XLdif=np.roll(XL,-1)-XL
    XLdif=XLdif[0:-1]
    if (len(XL) == 1):
      self.dx=0
      self.XL=XL
      self.Nx=1
    elif sum(np.round(XLdif[0], 10)==np.round(XLdif, 10))==len(XLdif):
      self.dx=XLdif[0]
      self.XL=XL
      self.Nx=len(XL)
    else:
      print("dx is not constant!")
    #list the y-coordinates of the data points, in increasing order
    YL=np.unique(self.df['Y'])
    YLdif=np.roll(YL,-1)-YL
    YLdif=YLdif[0:-1]
    if (len(YL) == 1):
      self.dy=0
      self.YL=YL
      self.Ny=1
    elif sum(np.round(YLdif[0], 10)==np.round(YLdif, 10))==len(YLdif):
      self.dy=YLdif[0]
      self.YL=YL
      self.Ny=len(YL)
    else:
      print("dy is not constant!")
    #list the z-coordinates of the data points, in increasing order
    ZL=np.unique(self.df['Z'])
    ZLdif=np.roll(ZL,-1)-ZL
    ZLdif=ZLdif[0:-1]
    if (len(ZL) == 1):
      self.dz=0
      self.ZL=ZL
      self.Nz=1
    elif sum(np.round(ZLdif[0], 10)==np.round(ZLdif, 10))==len(ZLdif):
      self.dz=ZLdif[0]
      self.ZL=ZL
      self.Nz=len(ZL)
    else:
      print("dz is not constant!")
    #make the meshgrid X, Y that will be useful for plotting
    [self.X,self.Y]=np.meshgrid(XL,YL)

  def generate_flattened_jones(self):
    '''
    Calculates the cumulative Jones matrices for each spatial position in the
        DataFrame, considering the entire depth (z-direction) of the sample.
    '''
    xs = self.df['X'].values
    ys = self.df['Y'].values

    newX=np.empty((self.Nx * self.Ny, 1, 1))
    newY=np.empty((self.Nx * self.Ny, 1, 1))

    for k in self.ks:
      jones_tot = np.empty((self.Nx * self.Ny, 2, 2), dtype=complex)

      jones = np.stack(self.df[k].values)

      start_time = time.time()

      for i in range(0, self.Nx):
        for j in range(0, self.Ny):
          column = jones[i*(self.Nz*self.Ny)+j*self.Nz:i*(self.Nz*self.Ny)+j*self.Nz+self.Nz-1]
          start_time2 = time.time()

          jones_tot[i*self.Ny + j] = multi_dot(column)
          newX[i*self.Ny + j]=xs[i*(self.Nz*self.Ny)+j*self.Nz]
          newY[i*self.Ny + j]=ys[i*(self.Nz*self.Ny)+j*self.Nz]

      end_time = time.time()
      elapsed_time = end_time - start_time
      print(f"k = {k} took {elapsed_time} seconds to run.")

      #XL/YL list of x/y coordinates, unique ! This list do not have same
      if self.ks[0] == k:
        self.flattened_df = pd.DataFrame({'X': newX.flatten(), 'Y': newY.flatten(), k: list(jones_tot)})
      else:
        self.flattened_df[k] = list(jones_tot)

  def help_compute_n_effective(self):
    """
    Computes the effective refractive index for light propagating through the
        sample, based on the orientation of the liquid crystal molecules
        and the extraordinary and ordinary refractive indices.
    """
    return ((self.n_para * self.n_perp)
            / np.sqrt(self.n_para ** 2 * np.cos(self.df['theta']) ** 2
            + self.n_perp ** 2 * np.sin(self.df['theta']) ** 2))

  def rotate_matrix(self, mat, theta, size = None, point = None):
    """
    Rotates a matrix or list of matrices by theta or a list of thetas
    """
    if point is not None:
      cos_theta, sin_theta = np.cos(theta), np.sin(theta)
      rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

      mat = np.array(mat).reshape(self.Nx, self.Ny)
      for i in range(mat.shape[0]):
          for j in range(mat.shape[1]):
              translation_matrix = np.array([[1, 0, -point[0]], [0, 1, -point[1]], [0, 0, 1]])
              reverse_translation_matrix = np.array([[1, 0, point[0]], [0, 1, point[1]], [0, 0, 1]])
              extended_matrix = np.eye(3)
              extended_matrix[:2, :2] = mat[i, j]

              transformed = reverse_translation_matrix @ rotation_matrix @ translation_matrix @ extended_matrix
              mat[i, j] = transformed[:2, :2]

      return mat

    if size is not None:
      rotation_matrices = np.empty([size, 2, 2])
      rotation_matrices[:, 0, 0] = np.cos(theta)
      rotation_matrices[:, 0, 1] = -np.sin(theta)
      rotation_matrices[:, 1, 0] = np.sin(theta)
      rotation_matrices[:, 1, 1] = np.cos(theta)
      rotation_matrices_inv = np.linalg.inv(rotation_matrices)

      return list(rotation_matrices @ mat @ rotation_matrices_inv)

    if mat.shape == (2, 1):
      rotation_matrices = np.empty([2, 2])
      rotation_matrices[0, 0] = np.cos(theta)
      rotation_matrices[0, 1] = -np.sin(theta)
      rotation_matrices[1, 0] = np.sin(theta)
      rotation_matrices[1, 1] = np.cos(theta)

      return rotation_matrices @ mat

    elif all(isinstance(x, np.ndarray) for x in mat) or mat.shape == (_, 2, 2):
      rotation_matrices = np.empty([len(mat), 2, 2])
      rotation_matrices[:, 0, 0] = np.cos(theta)
      rotation_matrices[:, 0, 1] = -np.sin(theta)
      rotation_matrices[:, 1, 0] = np.sin(theta)
      rotation_matrices[:, 1, 1] = np.cos(theta)
      rotation_matrices_inv = np.linalg.inv(rotation_matrices)

      return list(rotation_matrices @ mat @ rotation_matrices_inv)

    elif isinstance(mat, np.ndarray):
      rotation_matrices = np.empty([2, 2])
      rotation_matrices[0, 0] = np.cos(theta)
      rotation_matrices[0, 1] = -np.sin(theta)
      rotation_matrices[1, 0] = np.sin(theta)
      rotation_matrices[1, 1] = np.cos(theta)

      return rotation_matrices @ mat @ rotation_matrices_inv

    else:
      return "rotate_matrix: Invalid Input"

  def generate_jones(self):
    """
    Constructs the Jones matrices for light propagation through the liquid
        crystal sample.
    """
    for k in self.ks:
      jm_columns = pd.DataFrame()

      jones_matrices = []

      exp_term = np.exp(1j * k * (self.help_compute_n_effective() - self.n_perp) * self.dz)

      jones_matrices = np.zeros((exp_term.shape[0], 2, 2), dtype=complex)
      jones_matrices[:, 0, 0] = exp_term
      jones_matrices[:, 1, 1] = 1

      self.df[k] = self.rotate_matrix(jones_matrices, self.df['phi'].values)

  def propagate_field(self,
                      Ein = np.array([[1], [0]], dtype=complex),
                      rotated_sample_angle = 0, rotated_cross_pol_angle = 0,
                      rotated_pol_angle = 0):
    '''
    Computes the light vector after passing through the polarizer and
        the LC sample.
    Inputs:
      (optional) Ein: The incident electric field vector.
      (optional) rotated_sample_angle: the amount the sample is rotated, in degrees
    '''
    Ein = self.rotate_matrix(Ein, rotated_cross_pol_angle)

    for k in self.ks:
      jones_tot=np.stack(self.flattened_df[k].values)
      rotated_sample_matrices = self.rotate_matrix(jones_tot, rotated_sample_angle)
      self.LCout= np.matmul(rotated_sample_matrices, Ein)

  def filter_field(self, cross_pol = np.array([[0, 0], [0, 1]], dtype=complex),
                   rotated_cross_pol_angle=0):
    '''
    Computes the light vector after the analyzer.
    Inputs:
      (optional) cross_pol: The analyzer's matrix.
    '''
    cross_pol = self.rotate_matrix(cross_pol, rotated_cross_pol_angle, size = len(self.LCout))

    self.Eout = np.matmul(cross_pol, self.LCout)

  def compute_intensity(self):
    '''
    Computes the intensity of light after the analyzer.
    '''
    intensities = []

    for k in self.ks:
      Intensity_vector=np.abs(self.Eout[:,0,0],dtype=np.float64)**2+np.abs(self.Eout[:,1,0],dtype=np.float64)**2
      intensities.append(Intensity_vector)
    self.average_Is = [sum(values) / len(values) for values in zip(*intensities)]

  def plot_intensity_image(self, Ein = np.array([[1], [0]],dtype=complex),
                           cross_pol = np.array([[0, 0], [0, 1]], dtype=complex),
                           rotated_sample_angle=0, rotated_cross_pol_angle=0,
                           rotated_pol_angle=0, mypoint = (0, 0)):
    '''
    Generates a grayscale image representing the intensity distribution of light
        after interacting with the liquid crystal sample.
    '''
    for k in self.ks:
      self.propagate_field(Ein = np.array([[1], [0]],dtype=complex),
                          rotated_sample_angle=rotated_sample_angle,
                          rotated_cross_pol_angle=rotated_cross_pol_angle,
                          rotated_pol_angle = rotated_pol_angle)
      self.filter_field(cross_pol = np.array([[0, 0], [0, 1]], dtype=complex),
                        rotated_cross_pol_angle=rotated_cross_pol_angle)
      self.compute_intensity()

    #I=np.zeros((self.Nx,self.Ny))
    I2D=np.reshape(self.average_Is,(self.Nx,self.Ny))
    # Rotate the matrix by 90° CCW
    rotated_I2D = np.rot90(I2D, k=1)
    #print(I2D)
    Imin=np.min(I2D)
    Imax=np.max(I2D)
    print(Imax)
    #fig, a = plt.subplots()
    plt.imshow(rotated_I2D, interpolation='bicubic', cmap='gray', vmin=Imin, vmax=Imax)
    #plt.colorbar()  # Add a colorbar to show the intensity scale
    plt.grid(False)
    plt.axis('off')
    #a.plot(y, x)  # Switch x and y to rotate 90° CCW
    #plt.title('Intensity Image in Grayscale')
    #a.plot(y, x, color='red')  # Plot custom axes on top of the image
    plt.savefig('XPOL_image.png', transparent=False, bbox_inches='tight',
                pad_inches=0, format='png')
    plt.show()

  def plot_intensity_video(self):
    '''
    Generates a grayscale video representing the intensity distribution of light
        after interacting with the liquid crystal sample.
    '''
    frames = np.linspace(0, 2*math.pi, 360)

    intensity_frames = []

    for frame in frames:
      self.propagate_field(rotated_cross_pol_angle = frame)
      self.filter_field(rotated_cross_pol_angle = frame)
      self.compute_intensity()
      intensity_frames.append(np.array(self.average_Is).reshape(self.Nx, self.Ny))

    first_frame = np.rot90(intensity_frames[0], k=0)

    fig, a = plt.subplots()

    Imin=np.min(first_frame)
    Imax=np.max(first_frame)

    if (abs(Imin - Imax) < .001):
      Imin = 0
      Imax = 1

    frame = first_frame
    im = a.imshow(frame, interpolation='bicubic', cmap='gray', vmin=0,
                   vmax=.001)

    def update(frame_num):
      frame = np.rot90(intensity_frames[frame_num], k=0)
      im.set_array(frame)

      # Imin=np.min(frame)
      # Imax=np.max(frame)
      # im.set_norm(Normalize(vmin=Imin, vmax=Imax))

      a.set_title('Angle = ' + str(round(frame_num)))
      return [im]

    # Create an animation
    ani = FuncAnimation(fig, update, frames=359, blit=True)

    # Save the animation
    ani.save('dynamic_imshow' + str(time.strftime("%H:%M:%S", time.localtime())) +
             '.mp4', writer='ffmpeg', fps=30)

  def plot_slice_of_director_field(self, slice_normal='Z', slice_position=0.5,
                                   sampling_interval=5):
    '''
    Visualizes the orientation (director field) of the liquid crystal molecules
        within a specified slice of the sample.
    Inputs:
      slice_normal: The normal vector to the slicing plane.
      slice_position: The position of the slice along the normal vector,
          normalized between 0 and 1.
      sampling_interval: The interval between data points in the plotted slice,
          used to thin the data for clarity.
    '''
    df = self.get_df()
    sampling_interval_rows=sampling_interval
    sampling_interval_columns=sampling_interval
    #slice position a number between 0 and 1 0 zmin 1 zmax
    #select from the data frame the good slice
    if slice_normal=='Z':
      z_index=round(slice_position*self.Nz)
      z_coordinate=self.ZL[z_index]
      selected_rows = df[df[slice_normal] == z_coordinate]
      selected_rows = selected_rows.sort_values(by=['X', 'Y','Z'])
    else:
      print('Slice normal input is wrong')
    #making it 2D works only for zslice
    theta2D=np.reshape(selected_rows['theta'].values,(self.Nx,self.Ny))
    phi2D=np.reshape(selected_rows['phi'].values,(self.Nx,self.Ny))
    X2D=np.reshape(selected_rows['X'].values,(self.Nx,self.Ny))
    Y2D=np.reshape(selected_rows['Y'].values,(self.Nx,self.Ny))
    U2D=np.sin(theta2D) * np.cos(phi2D)
    V2D=np.sin(theta2D) * np.sin(phi2D)
    #coarsening the mesh
    X2Dplot=X2D[::sampling_interval_rows, ::sampling_interval_columns]
    Y2Dplot=Y2D[::sampling_interval_rows, ::sampling_interval_columns]
    U2Dplot=U2D[::sampling_interval_rows, ::sampling_interval_columns]
    V2Dplot=V2D[::sampling_interval_rows, ::sampling_interval_columns]
    phase_plot = phi2D[::sampling_interval, ::sampling_interval]

    fig, a = plt.subplots()
    color_map = cm.hsv(phase_plot.flatten())
    q1 = a.quiver(X2Dplot, Y2Dplot, U2Dplot, V2Dplot, headlength=0,
                   headwidth=0, headaxislength=0)
    q2 = a.quiver(X2Dplot, Y2Dplot, -U2Dplot, -V2Dplot, headlength=0,
                   headwidth=0, headaxislength=0)

    a.set_xlabel("X position")
    a.set_ylabel("Y position")
    # Set equal aspect ratio
    a.set_aspect('equal')
    plt.show()

  def set_df(self, new_df):
    self.df = new_df

  def get_df(self):
    '''
    Returns the DataFrame associated with the current instance of
        MyDataFrameProcessor.
    '''
    return self.df

  def get_flat_df(self):
    '''
    Returns the flattened DataFrame associated with the current instance of
        MyDataFrameProcessor.
    '''
    return self.flattened_df