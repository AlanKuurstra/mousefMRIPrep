import os

def split_exts(original_filename):
    #filename_part = os.path.basename(original_filename)
    filename_part = original_filename #want full path for derivatives
    exts = []
    ext_part = 'ext'
    while ext_part != '':
        filename_part, ext_part = os.path.splitext(filename_part)
        exts.append(ext_part)
    exts.reverse()
    exts = ''.join(exts)
    return filename_part, exts