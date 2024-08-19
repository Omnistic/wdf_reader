import matplotlib.pyplot as plt
import numpy as np
import pprint
import struct

VALID_SIGNATURE = ['WDF1']
SUPPORTED_VERSIONS = [0, 1]
VALID_BLOCK_SIZE = [512]

SCAN_TYPES = [
    'Unspecified',
    'Static',
    'Continuous',
    'StepRepeat',
    'FilterScan',
    'FilterImage',
    'StreamLine',
    'StreamLineHR',
    'PointDetector'
]

MEASUREMENT_TYPES = [
    'Unspecified',
    'Single',
    'Series',
    'Map'
]

DATA_TYPES = [
    'Arbitrary',
    'Spectral', # deprecated: use Frequency instead (spectral data type)
    'Intensity',
    'SpatialX', # X axis position
    'SpatialY', # Y axis position
    'SpatialZ', # Z axis (vertical) position
    'SpatialR', # rotary stage R axis position
    'SpatialTheta', # rotary stage theta angle
    'SpatialPhi', # rotary stage phi angle
    'Temperature',
    'Pressure',
    'Time',
    'Derived', # derivative type
    'Polarization',
    'FocusTrack', # focus track Z position
    'RampRate', # temperature ramp rate
    'Checksum',
    'Flags', # bit flags
    'ElapsedTime', # elapsed time interval
    'Frequency', # Frequency (such as wavelength or wavenumber).
    'Mp_Well_Spatial_X', # Microplate mapping origin X
    'Mp_Well_Spatial_Y', # Microplate mapping origin Y
    'Mp_LocationIndex', # Microplate mapping location index
    'Mp_WellReference', # Microplate mapping well reference
    'PAFZActual', # PAF focus distance from focus
    'PAFZError', # PAF distance between current and last positions,
    'PAFSignalUsed', # PAF signal used (0 = None, 1 = Top, 2 = Bottom, 3 = Correlation)
    'ExposureTime', # Measured exposure time per dataset
    'ExternalSignal', # Intensity data from external data source
    'Custom' # Custom data type: bit 31 is set
]

DATA_UNITS = [
    'Arbitrary',
    'RamanShift',
    'Wavenumber',
    'Nanometre',
    'ElectronVolt',
    'Micron',
    'Counts',
    'Electrons',
    'Millimetres',
    'Metres',
    'Kelvin',
    'Pascal',
    'Seconds',
    'Milliseconds',
    'Hours',
    'Days',
    'Pixels',
    'Intensity',
    'RelativeIntensity',
    'Degrees',
    'Radians',
    'Celsius',
    'Fahrenheit',
    'KelvinPerMinute',
    'FileTime',
    'Microseconds',
    'Volts',
    'Amps',
    'MilliAmps',
    'Strain',
    'Ohms',
    'DegreesR',
    'Coulombs',
    'PicoCoulombs'
]

def read_wdf(file_path):
    with open(file_path, mode='rb') as wdf:
        wdf_bytes = wdf.read()

    signature, version, block_size = struct.unpack('<4sIQ', wdf_bytes[0:16])
    signature = signature.decode('utf-8')

    if signature not in VALID_SIGNATURE or version not in SUPPORTED_VERSIONS or block_size not in VALID_BLOCK_SIZE:
        print('ERROR: {} does not use a recognised WDF format / version (signature: {} | version: {} | size: {}).'.format(file_path, signature, version, block_size))
        return False
    
    wdf_header = get_header(wdf_bytes)

    wdf_spectra, wdf_x_axes, wdf_x_axes_labels = get_spectra(wdf_bytes, wdf_header)

    return wdf_header, wdf_spectra, wdf_x_axes, wdf_x_axes_labels

def get_header(wdf_bytes):
    wdf_header = {}

    wdf_header['Title'] = struct.unpack('<160s', wdf_bytes[240:400])[0].decode('utf-8').rstrip('\x00')
    wdf_header['Username'] = struct.unpack('<32s', wdf_bytes[208:240])[0].decode('utf-8').rstrip('\x00')
    wdf_header['MeasurementType'] = MEASUREMENT_TYPES[struct.unpack('<I', wdf_bytes[132:136])[0]]
    wdf_header['ScanType'] = SCAN_TYPES[struct.unpack('<I', wdf_bytes[128:132])[0]]
    wdf_header['LaserWavenumber'] = struct.unpack('<f', wdf_bytes[160:164])[0]
    wdf_header['Count'] = struct.unpack('<Q', wdf_bytes[72:80])[0]
    wdf_header['SpectraUnit'] = DATA_UNITS[struct.unpack('<I', wdf_bytes[152:156])[0]]
    wdf_header['PointsPerSpectrum'] = struct.unpack('<I', wdf_bytes[60:64])[0]
    wdf_header['DataOriginCount'] = struct.unpack('<I', wdf_bytes[92:96])[0]
    wdf_header['Capacity'] = struct.unpack('<Q', wdf_bytes[64:72])[0]
    wdf_header['ApplicationName'] = struct.unpack('<24s', wdf_bytes[96:120])[0].decode('utf-8').rstrip('\x00')
    wdf_header['ApplicationVersion'] = '.'.join(map(str, struct.unpack('<4H', wdf_bytes[120:128])))
    wdf_header['XListLength'] = struct.unpack('<I', wdf_bytes[88:92])[0]
    wdf_header['YListLength'] = struct.unpack('<I', wdf_bytes[84:88])[0]
    wdf_header['AccumulationCount'] = struct.unpack('<I', wdf_bytes[80:84])[0]

    return wdf_header

def get_spectra(wdf_bytes, wdf_header):
    data_block_start_index = wdf_bytes.find(b'DATA')
    xlst_block_start_index = wdf_bytes.find(b'XLST')

    x_axes = []
    x_axes_labels = []   
    spectra = []

    for spectrum_index in range(wdf_header['Count']):
        spectrum_data_index = data_block_start_index + 16 + (spectrum_index * wdf_header['PointsPerSpectrum'] * 4)
        xlst_data_index = xlst_block_start_index + 24

        spectra.append(struct.unpack('<{}f'.format(wdf_header['PointsPerSpectrum']), wdf_bytes[spectrum_data_index:spectrum_data_index + (wdf_header['PointsPerSpectrum'] * 4)]))
        x_axes.append(struct.unpack('<{}f'.format(wdf_header['XListLength']), wdf_bytes[xlst_data_index:xlst_data_index + (wdf_header['XListLength'] * 4)]))
        x_axes_labels.append({
            'XListType': DATA_TYPES[struct.unpack('<I', wdf_bytes[xlst_block_start_index + 16:xlst_block_start_index + 20])[0]],
            'XListUnits': DATA_UNITS[struct.unpack('<I', wdf_bytes[xlst_block_start_index + 20:xlst_block_start_index + 24])[0]],
        })

    return spectra, x_axes, x_axes_labels

if __name__ == '__main__':
    wdf_header, wdf_spectra, wdf_x_axes, wdf_x_axes_labels = read_wdf(r'C:\Users\davnguyen\Documents\python_projects\raman\whole_range.wdf')

    pprint.pp(wdf_header)
    pprint.pp(wdf_x_axes_labels)

    plt.figure()
    plt.plot(wdf_x_axes[0], wdf_spectra[0])
    plt.xlabel('{} ({})'.format(wdf_x_axes_labels[0]['XListType'], wdf_x_axes_labels[0]['XListUnits']))
    plt.show()
