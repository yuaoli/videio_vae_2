import os
import pydicom
import numpy as np
import nibabel as nib
import tarfile
from pydicom.errors import InvalidDicomError

class DicomDisContinuous(Exception):
    pass

def read_dicom_from_tar_gz(tar_gz_path):
    dicom_files = []   
    # 打开 tar.gz 文件
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # 遍历压缩包内的文件
        for member in tar.getmembers():
            if member.isfile():
                # 提取文件内容并读取为 DICOM 格式
                file = tar.extractfile(member)
                try:
                    dicom_file = pydicom.dcmread(file)
                    dicom_files.append(dicom_file)
                except InvalidDicomError:
                    continue  # 如果文件不是 DICOM 文件，则跳过
    return dicom_files

def load_dicom_series(directory, check_continuous=True):
    dicom_files = []
    if 'tar.gz' in directory:
        dicom_files = read_dicom_from_tar_gz(directory)
    else:
        for root, dirs, f in os.walk(directory):
            for file in f:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    try:
                        dicom_file = pydicom.dcmread(file_path)
                        if 'SAG' in dicom_file.ProtocolName: 
                            dicom_files.append(dicom_file)
                    except pydicom.errors.InvalidDicomError:
                        continue
    dicom_files.sort(key=lambda x: x.InstanceNumber)
    # check continuous
    if check_continuous:
        inst_nums = [x.InstanceNumber for x in dicom_files]
        miss = np.setdiff1d(np.arange(min(inst_nums), max(inst_nums)+1), inst_nums)
        if len(miss) > 0:
            raise DicomDisContinuous(f"instance number is not continuous: missing {miss}")
    # todo: check slope etc. if it is for CT
    pixel_data = np.stack([df.pixel_array for df in dicom_files])  # DHW
    # to WHD
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    return dicom_files, pixel_data


def get_affine(dicom_files):
    ds_f0 = dicom_files[0]
    image_position = ds_f0.ImagePositionPatient
    image_orientation = np.array(ds_f0.ImageOrientationPatient).reshape(2, 3).astype(np.float32)
    # row cosine, col cosine
    sp_row, sp_col = ds_f0.PixelSpacing
    normal = np.cross(image_orientation[0], image_orientation[1])
    projs = [np.dot(normal, _.ImagePositionPatient) for _ in dicom_files]
    slice_interval = np.mean(np.diff(projs))
    # np.dot(orientation_k, image_position)
    T = np.eye(4)
    T[:3, 0] = image_orientation[0] * sp_row
    T[:3, 1] = image_orientation[1] * sp_col
    T[:3, 2] = normal * slice_interval
    T[:3, -1] = image_position
    # why? to RAP
    T[1] *= -1
    T[0] *= -1
    return T


def add_version_info(nifti_img, series_id=None, instance_range=None):
    header = nifti_img.header
    ver_info = f"V:knee1.0;DO:whd"  # V: version, DO: data order
    assert len(ver_info) < 24
    header["aux_file"] = ver_info

    dcm_info = []
    if series_id:
        assert len(series_id) <= 64, f"maximum length exceed {len(series_id)}"
        dcm_info.append(f"S:{series_id}")
    if instance_range:
        dcm_info.append("R:" + "-".join([str(_) for _ in instance_range]))
    assert len(ver_info) < 80
    header["descrip"] = ";".join(dcm_info)
    return nifti_img


def convert_to_nifti(dicom_files, pixel_data, output_file=None):
    affine = get_affine(dicom_files)
    instance_range = [dicom_files[0].InstanceNumber, dicom_files[-1].InstanceNumber]
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nifti_img = add_version_info(nifti_img, series_id=dicom_files[0].SeriesInstanceUID,
                                 instance_range=instance_range)
    return nifti_img
    # nib.save(nifti_img, output_file)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_src', '-s', type=str, default="", required=True)
    parser.add_argument('--nii_dst', '-d', type=str, default="", required=True)
    parser.add_argument('--depth', type=int, default=3, required=False)
    parser.add_argument('--min_slices', type=int, default=5, required=False)
    return parser.parse_args()


def test():
    series = [
        "/Users/dw/Downloads/lumbar_disc/3606018/1.2.156.112605.14038010223091.20230531234223.2.4828.3/1.2.156.112605.14038010223091.20230531234223.3.4828.4"
    ]
    for dicom_dir in series:
        output_file = os.path.basename(dicom_dir) + ".nii.gz"
        print(output_file)
        dicom_files, pixel_data = load_dicom_series(dicom_dir)
        convert_to_nifti(dicom_files, pixel_data, output_file)


def main():
    args = parse_args()
    from pathlib import Path
    from tqdm import tqdm
    depth = args.depth
    folders = list(Path(args.dcm_src).glob("*/" * depth))
    # if depth == 3:
    #     folders = list(Path(args.dcm_src).glob("*/*/*"))
    # elif depth == 2:
    #     folders = list(Path(args.dcm_src).glob("*/*"))
    # folders = glob.glob(os.path.join(args.dcm_src, "*/*/*"))
    for dicom_dir in tqdm(folders):
        try:
            dicom_dir = str(dicom_dir)
            sub = "/".join(dicom_dir.split("/")[-depth:])
            output_file = Path(args.nii_dst) / (sub + ".nii.gz")
            if output_file.exists():
                continue
            output_file.parent.mkdir(parents=True, exist_ok=True)
            dicom_files, pixel_data = load_dicom_series(dicom_dir)
            if len(dicom_files) < args.min_slices:
                print(f"skip {dicom_dir}, n_slices={len(dicom_files)} < {args.min_slices}")
                continue
            convert_to_nifti(dicom_files, pixel_data, str(output_file))
        except Exception as ex:
            if isinstance(ex, DicomDisContinuous):
                print(ex)
            else:
                print(ex)
                #import pdb; pdb.set_trace()


if __name__ == '__main__':
    # test()
    main()
