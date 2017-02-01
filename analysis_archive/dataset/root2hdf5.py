def __destructure(structured, dtype=np.float64):
    return structured.view(dtype).reshape(structured.shape + (-1,))

 array = root_numpy.root2array(
                glob.glob('{}/{}/*'.format(tmpdir, group))
            )


def root2array(input_path, dtype=np.float64):
    structured = root_numpy.root2array(input_path)
    return structured.view(dtype).reshape(structured.shape + (-1,))
