import numpy as np
import subprocess as sp
import os
import tempfile

BART_PATH = os.environ.get('BART_PATH')
if BART_PATH is None:
    raise ValueError("BART_PATH not set")

def readcfl(name):
    """
    read cfl file
    """

    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

def writecfl(name, array):
    """
    write cfl file
    """
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def check_out(cmd, split=True):
    """ utility to check_out terminal command and return the output"""

    strs = sp.check_output(cmd, shell=True).decode()

    if split:
        split_strs = strs.split('\n')[:-1]
    return split_strs

def bart(nargout, cmd, *args, return_str=False):
    """
    Call bart from the system command line.

    Args:
        nargout (int): The number of output arguments expected from the command.
        cmd (str): The command to be executed by bart.
        *args: Variable number of input arguments for the command.
        return_str (bool, optional): Whether to return the output as a string. Defaults to False.

    Returns:
        list or str: The output of the command. If nargout is 1, returns a single element list.
                     If return_str is True, returns the output as a string.

    Raises:
        Exception: If the command exits with an error.

    Usage:
        bart(<nargout>, <command>, <arguments...>)
    """
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return None

    name = tempfile.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = BART_PATH + ' ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = sp.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print("Make sure bart is properly installed")
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

def getname(dataset, contrast, cplx, id_):
    return "%s_%s_%s_%d"%(dataset, contrast, cplx, id_)