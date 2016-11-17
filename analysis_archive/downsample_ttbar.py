from subprocess import call
import os
import ROOT

def sh(cmd):
    call(cmd, shell=True)

def count(rfile):
    tfile = ROOT.TFile(rfile)
    ttree = tfile.Get('NNinput')
    return ttree.GetEntries()
    
def hadd(dirpath):
    out = os.path.basename(dirpath)
    sh('hadd {} {}/*'.format(out, dirpath))
    return out, count(out)

def downsample(path, n):
    out = path.replace('.root', '.downsampled.root')
    sh('downsample --infile {} --outfile {} --nkeep {}'.format(path, out, n))


p1, c1 = hadd('/lcg/storage15/atlas/gagnon/data/NNinput_1/user.lgagnon.410000.ttbar.NNinput_1.e3698_s2608_s2183_r7725_r7676_p2666_tag2.4.15-2-0_2_v0_NNinput_1.root')

p2, c2 = hadd('/lcg/storage15/atlas/gagnon/data/NNinput_1/user.lgagnon.407009.ttbar.NNinput_1.e4023_s2608_r7725_r7676_p2666_tag2.4.15-2-0_2_v0_NNinput_1.root')

p3, c3 = hadd('/lcg/storage15/atlas/gagnon/data/NNinput_1/user.lgagnon.407010.ttbar.NNinput_1.e4023_s2608_r7725_r7676_p2666_tag2.4.15-2-0_2_v0_NNinput_1.root')

p4, c4 = hadd('/lcg/storage15/atlas/gagnon/data/NNinput_1/user.lgagnon.407011.ttbar.NNinput_1.e4023_s2608_r7725_r7676_p2666_tag2.4.15-2-0_2_v0_NNinput_1.root')

total = c1+c2+c3+c4
print total
fkeep = 70000.0/total
print fkeep
n1 = int(round(fkeep * c1))
print n1
n2 = int(round(fkeep * c2))
print n2
n3 = int(round(fkeep * c3))
print n3
n4 = int(round(fkeep * c4))
print n4

downsample(p1, n1)
downsample(p2, n2)
downsample(p3, n3)
downsample(p4, n4)
       

