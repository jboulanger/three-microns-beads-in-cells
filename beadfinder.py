from typing import List

import math
import numpy as np
from scipy.ndimage import white_tophat, maximum_filter
from scipy.special import jn

import pandas as pd

from cellpose import models, core

import matplotlib.pyplot as plt
import matplotlib.patches

from pathlib import Path
import nd2
import tifffile
import yaml


def compute_otf(
    shape: List[int],
    spacing: List[float],
    numerical_aperture: float,
    medium_refactive_index: float,
    wavelength: float,
):
    """Basic defocus optical transfer function for wide field

    Parameters
    ----------
    shape : list[int]
        image size [D,H,W]
    spacing: list [float]
        spacing [D,H,W] in 3D in physical unit
    numerical_aperture: float
        objective numerical aperture
    medium_refractive_index: float
        the refractive index of the medium
    wavelength: float
        the emission wavelength in unit

    Returns
    -------
    otf : np.ndarray(dtype=np.complex128)
        The optical transfer function
    """
    a = (numerical_aperture / wavelength) ** 2
    b = (medium_refactive_index / wavelength) ** 2
    kx = np.fft.fftfreq(shape[2], spacing[2]).reshape([1, 1, shape[2]])
    ky = np.fft.fftfreq(shape[1], spacing[1]).reshape([1, shape[1], 1])
    z = (
        np.concatenate(
            (np.arange(0, shape[0] // 2), np.arange(-shape[0] // 2, 0))
        ).reshape([shape[0], 1, 1])
        * spacing[0]
    )
    d2 = kx**2 + ky**2
    W = z * np.sqrt(np.maximum(0, b - d2))
    P = (d2 < a).astype(float)
    psf = np.square(np.abs(np.fft.ifft2(P * np.exp(2j * math.pi * W))))
    psf = psf / psf.sum()
    otf = np.fft.fftn(psf)
    return otf


def compute_pinhole_otf(shape, spacing, diameter):
    """Fourier transform of a circular aperture

    $(2 J_1(k) / k)$
    Note
    ----
    THis is not the Airy disk
    """
    kx = np.fft.fftfreq(shape[2], spacing[2]).reshape([1, 1, shape[2]])
    ky = np.fft.fftfreq(shape[1], spacing[1]).reshape([1, shape[1], 1])
    k = math.pi / 2 * diameter * np.sqrt(kx**2 + ky**2)
    plane = np.zeros([1, shape[1], shape[2]])
    plane[k > 1e-8] = 2 * jn(1, k[k > 1e-8]) / k[k > 1e-8]
    plane[0, 0, 0] = 1
    pinhole = np.zeros(shape)
    pinhole[0] = plane
    return pinhole


def compute_confocal_otf(
    shape,
    spacing,
    numerical_aperture=1.4,
    medium_refractive_index=1.35,
    wavelength_ex=0.5,
    wavelength_em=0.5,
    pinhole=0.5,
):
    """Compute the optical transfer function of a confocal microscope"""
    otf1 = compute_otf(
        shape, spacing, numerical_aperture, medium_refractive_index, wavelength_ex
    )
    otf2 = compute_otf(
        shape, spacing, numerical_aperture, medium_refractive_index, wavelength_em
    )
    pinhole = compute_pinhole_otf(shape, spacing, pinhole)
    psf1 = np.real(np.fft.ifftn(otf1 * pinhole))
    psf2 = np.real(np.fft.ifftn(otf2))
    otf = np.fft.fftn(psf1 * psf2)
    return otf


def compute_shell_fft(shape: List[int], spacing: List[float], diameter: float):
    """fft of a 3D shell

     Parameters
    ----------
    shape : list
        image size [D,H,W]
    spacing: list
        spacing in 3D in physical unit [D,H,W]
    diameter: float
        diameter of the spherical shell

    Returns
    -------
    shell : np.ndarray(dtype=np.complex128)
        The Fourier transform of a 3D shell.
    """
    kx = np.fft.fftfreq(shape[2], spacing[2]).reshape([1, 1, shape[2]])
    ky = np.fft.fftfreq(shape[1], spacing[1]).reshape([1, shape[1], 1])
    kz = np.fft.fftfreq(shape[0], spacing[0]).reshape([shape[0], 1, 1])
    shell = np.sinc(diameter * np.sqrt(kx**2 + ky**2 + kz**2))
    return shell


def create_sphere(shape, spacing, diameter, thickness, centers=None):
    """Create spherical shells centered at the given centers with

    Parameter
    ---------
    shape : size of the array
    pixel_size : dimension of the pixels along the dimension of the array
    diameter : diameter of the sphere in unit
    thickness : thicknes of the sphere in unit
    centers : centers coordinates in unit
    """
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")

    if centers is None:
        # center = [[0.5 * p * n for n,p in zip(shape, pixel_size)]]

        d = np.sqrt(sum([((x - n / 2) * p) ** 2 for x, n, p in zip(g, shape, spacing)]))

        out = np.exp(-0.5 * np.square(np.abs(d - diameter / 2) / thickness)) + 0.5 * (
            d < diameter / 2
        )
    else:
        out = np.zeros(shape)
        for ck in centers:
            d = sum([(p * x - c) ** 2 for x, p, c in zip(g, spacing, ck)])
            d = np.abs(np.sqrt(d) - diameter / 2)
            out = out + np.exp(-0.5 * np.square(d / thickness))
    return out


def create_confocal_bead(
    shape,
    spacing,
    diameter,
    thickness,
    numerical_aperture,
    medium_refractive_index,
    wavelength_ex,
    wavelength_em,
    pinhole,
):
    """Create a spherical shell observed in confocal microscopy"""
    otf1 = compute_otf(
        shape, spacing, numerical_aperture, medium_refractive_index, wavelength_ex
    )
    otf2 = compute_otf(
        shape, spacing, numerical_aperture, medium_refractive_index, wavelength_em
    )
    pinhole = compute_shell_fft(shape, spacing, pinhole)
    psf1 = np.real(np.fft.ifftn(otf1 * pinhole))
    psf2 = np.real(np.fft.ifftn(otf2))
    otf = np.fft.fftn(psf1 * psf2)
    s1 = compute_shell_fft(shape, spacing, diameter)
    s2 = compute_shell_fft(shape, spacing, thickness)
    b = otf * s1 * s2
    b = b / b.max()
    return b


def wiener(f, y, s):
    """Wiener filter
    Parameters
    ----------
    f : np.ndarray
        filter in Fourier space
    y : np.ndarray
        image
    s : float
        Regularization parameter

    Returns
    -------
    filtered image
    """
    return ftconv(np.conj(f) / (s + np.abs(f) ** 2), y)


def ftconv(f, x):
    """Convolution in Fourier space
    Parameter
    """
    return np.real(np.fft.ifftn(f * np.fft.fftn(x)))


def deconvloc(f, y, s, niter):
    x = wiener(f, y, 0.01)
    for _ in range(niter):
        x = x * ftconv(np.conj(f), y / np.maximum(0.1, ftconv(f, x)))
        x = np.maximum(0.0, x - s * x.max())
    return x


def label_spheres(shape, spacing, diameter, thickness=None, centers=None):
    """Measure intensity inside sphere

    Parameter
    ---------
    shape : size of the array
    spacing : dimension of the pixels along the dimension of the array
    diameter : radius of the sphere in unit
    thickness : thickness of the sphere in unit
    centers : centers coordinates in unit

    Result
    ------
    Numpy array with labelled spheres
    """
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    out = np.zeros(shape, dtype=np.uint32)
    for label, ck in enumerate(centers):
        d = np.sqrt(sum([(p * x - c) ** 2 for x, p, c in zip(g, spacing, ck)]))
        if thickness is None:
            out[d < diameter / 2] = label + 1
        else:
            d = np.abs(d - diameter / 2)
            out[d < thickness / 2] = label + 1
    return out


def normzxcorr(template: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Normalize zeros centered cross correlation

    Nomalize the image locally in the support of the template

    Parameters
    ----------
    template : template to match
    image : image

    Result
    ------
    normalized cross correlation map
    """
    mask = template > 0.1 * template.max()
    F = np.fft.fftn(image)
    T = np.conj(np.fft.fftn(template))
    A = np.conj(np.fft.fftn(mask))
    n = mask.sum()
    m = np.fft.fftshift(np.real(np.fft.ifftn(F * A))) / n
    delta = np.square(image - m)
    s = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(delta) * A))) / n
    z = (image - m) / np.sqrt(s)
    xc = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(z) * T))) / n
    return xc


def detect_beads(
    img,
    spacing,
    diameter,
    thickness,
    numerical_aperture,
    medium_refractive_index,
    wavelength_ex=0.5,
    wavelength_em=0.5,
    pinhole=0.5,
    sparsity=1e-3,
    num_iter=20,
    threshold=0.1,
):
    """Detect fluorescent beads and return coordinates"""

    # create a bead template
    bead_ft = create_confocal_bead(
        img.shape,
        spacing,
        diameter,
        thickness,
        numerical_aperture,
        medium_refractive_index,
        wavelength_ex,
        wavelength_em,
        pinhole,
    )

    # detect beads using a deconvolution approach
    # score = deconvloc(bead_ft, img, sparsity, num_iter)

    template = np.fft.fftshift(np.real(np.fft.ifftn(bead_ft)))
    score = normzxcorr(template, np.log(1 + img.astype(float) / img.max()))

    # find local maximas positions
    footprint = [diameter / n for n in spacing]
    zyx = np.argwhere(
        np.logical_and(
            score > score.max() * threshold, score == maximum_filter(score, footprint)
        )
    ).astype(float)

    # rescale the pixel coordinate to physical units
    for k in range(zyx.shape[0]):
        zyx[k] = zyx[k] * np.array(spacing)
    return zyx


def detect_spheres(
    img: np.ndarray, spacing, diameter: float, thickness: float, threshold: float = 0.5
) -> np.ndarray:
    """Detect spheres using normalized cross correlation

    Parameters
    ----------
    img : image
    spacing : list of pixel size
    diameter : value of the radius in physical unit
    thickness : thickness of the sphere in physical unit
    threshold : threshold

    Result
    ------
    An ndarray with the coordinates of the center of the sphere in zyx in
    physical units
    """

    # create the sphere template
    template = create_sphere(img.shape, spacing, diameter, thickness)

    # preprocess the image using a white_tophat
    tmp = white_tophat(img, 5).astype(float)

    #  comput the normlized cross correlation
    xc = normzxcorr(template, np.log(1.0 + tmp / tmp.max()))

    # find local maximas positions
    footprint = [diameter / n for n in spacing]
    zyx = np.argwhere(
        np.logical_and(xc > xc.max() * threshold, xc == maximum_filter(xc, footprint))
    ).astype(float)

    # rescale the pixel coordinate to physical units
    for k in range(zyx.shape[0]):
        zyx[k] = zyx[k] * np.array(spacing)
    return zyx


def labelprops(labels, intensity, spacing):
    """Compute statistics in labels for each channel

    Similar to regionprops but use all channels, take into account the spacing
    and returns a list of dictionnary.

    Parameters
    ----------
    label : array (D,H,W) with labels
    intensity   : array (D,C,H,W) with intensity
    spacing : list of pixel size in D,H,W axes.

    Return
    ------
    A list of dictionnary with label, volume and mean_intensity
    """
    import numpy.ma as ma

    label_list = [x for x in np.unique(labels) if x > 0]
    dv = np.prod(np.array(spacing))
    res = []
    for lab in label_list:
        # measure the mean intensity in each channel
        if labels.ndim == intensity.ndim:
            m = ma.array(intensity, mask=(labels != lab)).mean()
        else:
            m = [
                ma.array(intensity[:, c, :, :], mask=(labels != lab)).mean()
                for c in range(intensity.shape[1])
            ]
        # measure the volume
        v = (labels == lab).sum()
        res.append({"label": lab, "volume": v * dv, "mean intensity": m})
    return res


def process_img(
    img,
    spacing,
    cell_diameter=100,
    cell_cytosolic_channel=1,
    cell_nuclear_channel=3,
    cell_stitch_threshold=0.9,
    bead_diameter=3,
    bead_thickness=0.1,
    bead_threshold=0.5,
    numerical_aperture=1.4,
    medium_refractive_index=1.33,
    wavelength_ex=0.5,
    wavelength_em=0.5,
    pinhole=0.5,
    sparsity=1e-3,
    num_iter=20,
):
    """Detect beads inside cells

    Only beads at least 75% inside the cells are detected as inside

    Parameters
    ----------
    img : mp.ndarray (D,C,H,W)
    spacing  : size of pixels in [D,H,W] axes
    radius : radius of the beads

    Returns
    -------
    tbl_cells: pandas.DataFrame
        statistics per cell
    tbl_beads: pandas.DataFrame
        measurements per bead
    labels @ np.ndarray
        labels for the cells and beads
    """

    # segment the cells
    model = models.Cellpose(gpu=core.use_gpu(), model_type="cyto2")
    # median_filter(img[:, cell_cytosolic_channel], [3, 3, 3]),
    img_preprocessed = np.stack(
        (
            np.log(img[:, cell_cytosolic_channel].astype(float)),
            img[:, cell_nuclear_channel].astype(float),
        ),
        axis=1,
    )
    cell_labels = model.eval(
        img_preprocessed,
        diameter=cell_diameter,
        channels=[1, 2],
        do_3D=False,
        stitch_threshold=cell_stitch_threshold,
    )[0]

    cell_props = labelprops(cell_labels, img, spacing)

    centers = detect_spheres(
        img[:, 2], spacing, bead_diameter, bead_thickness, bead_threshold
    )

    # centers = detect_beads(
    #     img[:, 2],
    #     spacing,
    #     bead_diameter,
    #     bead_thickness,
    #     numerical_aperture,
    #     medium_refractive_index,
    #     wavelength_ex,
    #     wavelength_em,
    #     pinhole,
    #     sparsity,
    #     num_iter,
    #     bead_threshold,
    # )

    shape = [img.shape[n] for n in (0, 2, 3)]
    bead_labels = label_spheres(shape, spacing, bead_diameter, bead_thickness, centers)

    # filters beads inside a cell by at beads 75 percent of its (shell) volume
    frac_inside = [
        p["mean intensity"] for p in labelprops(bead_labels, cell_labels > 0, spacing)
    ]
    inside = np.array(frac_inside) > 0.75

    # get the indices of the cells for each bead as the label at the center
    bead_cell_idx = np.array(
        [cell_labels[tuple((p / spacing).astype(int))] for p in centers]
    )
    bead_cell_idx[np.logical_not(inside)] = 0

    # preprocess the intensity
    tmp = np.stack(
        [white_tophat(img[:, k, :, :], 7) for k in range(img.shape[1])], axis=1
    )

    # measure intensities
    bead_props = labelprops(bead_labels, tmp, spacing)

    # compute statistics per cells
    tbl_cells = pd.DataFrame(
        {
            "Label": [int(c["label"]) for c in cell_props],
            "Volume": [c["volume"] for c in cell_props],
            **{
                f"Mean_intensity_ch{k}": [c["mean intensity"][k] for c in cell_props]
                for k in range(img.shape[1])
            },
            "Num_beads": [np.sum(bead_cell_idx == c["label"]) for c in cell_props],
        }
    )

    # compute statistics per beads
    tbl_beads = pd.DataFrame(
        {
            "Label": [b["label"] for b in bead_props],
            "Volume": [b["volume"] for b in bead_props],
            "X": centers[:, 2],
            "Y": centers[:, 1],
            "Z": centers[:, 0],
            "R": [bead_diameter / 2] * len(bead_props),
            "Diameter": [bead_diameter] * len(bead_props),
            **{
                f"Mean_intensity_ch{k}": [b["mean intensity"][k] for b in bead_props]
                for k in range(img.shape[1])
            },
            "Cell": bead_cell_idx,
            "Fraction_inside": frac_inside,
        }
    )
    return tbl_cells, tbl_beads, np.stack([cell_labels, bead_labels])


def show_beads(img, spacing, beads):
    """Display the image and the beads as circles

    img: array (D,C,H,W) image to display
    spacing : size in micron
    beads : pandas dataframe with X,Y,R and Fraction_inside
    """

    ax = plt.gca()

    if img.ndim == 4:
        tmp = np.amax(img[:, 0:3, :, :] - img.min(), 0)
        tmp = np.moveaxis(tmp, [1, 2, 0], [0, 1, 2])
        ax.imshow(tmp)
    else:
        ax.imshow(np.amax(img, 0))

    x = beads["X"] / spacing[1]
    y = beads["Y"] / spacing[2]
    r = beads["R"] / spacing[2]
    c = ["g" if x > 0.9 else "r" for x in beads["Fraction_inside"]]
    for xi, yi, ri, ci in zip(x, y, r, c):
        p = matplotlib.patches.Circle(
            (xi, yi), radius=ri, fill=False, edgecolor=ci, linewidth=1
        )
        ax.add_patch(p)
    plt.axis("off")


def bead_control(img, spacing, beads, thickness):
    """Display the image and the beads as circles

    img: array (D,C,H,W) image to display
    spacing : size in micron
    beads : pandas dataframe with X,Y,R and Fraction_inside

    """
    x = beads["X"]
    y = beads["Y"]
    z = beads["Z"]
    d = beads["Diameter"].mean()
    centers = np.vstack([x, y, z]).T
    return create_sphere(img.shape, spacing, d, thickness, centers)


class BeadFinder:
    """Manage files and process"""

    def __init__(self, config_path: Path = None, src=None, dst=None):
        if config_path is not None:
            self.load_config(config_path)
        else:
            if src is not None:
                self.source = src
            if dst is not None:
                self.destination = dst

        print("Source folder accessible?", self.source.exists())
        if self.destination.exists() is False:
            print("Creating destination folder.")
            self.destination.mkdir(parents=True)
        print("Destination folder accessible?", self.destination.exists())

        self.scan_source_folder()
        nfiles = len(pd.unique(self.filelist["name"]))
        print(f"Discovered {len(self.filelist)} positions in {nfiles} files.")

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.source = Path(config["source"])
        self.destination = Path(config["destination"])

    def scan_source_folder(self):
        """List all files in the source folder"""
        files = self.source.glob("*.nd2")
        filelist = []
        for filepath in files:
            with nd2.ND2File(filepath) as f:
                if f.ndim == 4:
                    filelist.append(
                        {"folder": filepath.parent, "name": filepath.name, "fov": 0}
                    )
                else:
                    for k in range(f.shape[0]):
                        filelist.append(
                            {"folder": filepath.parent, "name": filepath.name, "fov": k}
                        )

        self.filelist = pd.DataFrame.from_records(filelist)

    def process_item(self, row: pd.DataFrame):

        with nd2.ND2File(self.source / row["name"]) as f:
            spacing = f.metadata.channels[0].volume.axesCalibration[::-1]
            spacing[0] = spacing[0] * 0.6
            # img = f.asarray(row["fov"])

            img = f.to_dask(False)
            img = img[row["fov"], :, :, :200, :200].compute()

        cells_df, beads_df, labels = process_img(
            img, spacing, cell_stitch_threshold=0.1
        )

        cells_df["name"] = row["name"]

        cname = row["name"].replace(".nd2", f'-{row["fov"]}-cells.csv')
        cells_df.to_csv(self.destination / cname)

        beads_df["fov"] = row["fov"]
        bname = row["name"].replace(".nd2", f'-{row["fov"]}-beads.csv')
        beads_df.to_csv(self.destination / bname)

        tname = row["name"].replace(".nd2", f'-{row["fov"]}-labels.tif')
        tifffile.imwrite(self.destination / tname, labels)

        return cells_df, beads_df

    def process_item_safe(self, row):
        """Process a row in the filelist"""
        try:
            cells_df, beads_df = self.process_item(row)
        except Exception as e:
            print("Error on file", row["name"], " position", row["fov"])
            print(e)
            return None, None
        return cells_df, beads_df

    def process_dataset(self):
        """Process the entire dataset sequentially"""
        results = [self.process_item(row) for row in self.filelist.iloc]
        return results
        # concatenate all results in 1 data frame
        cells = pd.concat([c for c, _ in results])
        beads = pd.concat([b for _, b in results])
        return cells, beads

    def process_parallel_dataset(self):
        """Process the entire dataset sequentially"""
        import dask

        # create tasks
        tsk = [dask.delayed(self.process_item_safe)(row) for row in self.filelist.iloc]

        # run the tasks
        results = dask.compute(tsk)

        # concatenate all results in 1 data frame
        cells = pd.concat([c for c, _ in results[0]])
        beads = pd.concat([b for _, b in results[0]])
        return cells, beads


def main():

    import argparse

    parser = argparse.ArgumentParser(description="beadfinder")
    parser.add_argument("--src", help="source folder", required=True)
    parser.add_argument("--dst", help="destination folder", required=True)
    args = parser.parse_args()
    bf = BeadFinder(src=args.src, dst=args.dst)
    bf.process_dataset()


if __name__ == "__main__":
    main()
