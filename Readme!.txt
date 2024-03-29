	---- SimRIS Channel Simulator v2.0. (CoreLab.ku.edu.tr/tools/SimRIS) ----

 Note: MIMO extension of SimRIS Channel Simulator is available now!

1- First, open the SimRIS_GUI.m and run it in MATLAB.

2- In the opened GUI, choose and select all parameters. Then, click the "Run SimRIS" button!
(Note: SimRIS needs a parallel computing toolbox!)

3- If the simulation is not executed, please check the Error Control Box!

4- If the simulation is succesfully executed, H (NxNtxNsym), G (NrxNxNsym), and D (NrxNtxNsym) can be directly used from the MATLAB workspace.

   N: Number of Transmit Reflectors, Nt: Number of Transmit Antennas, Nr: Number of Receive Antennas, Nsym: Number of Channel Realizations

5- Using the "Save as" button, the channels can be downloaded as a ".mat" format.


License
This code package is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
https://creativecommons.org/licenses/by-nc-sa/4.0/

Citation
If you in any way use this code for research that results in publications, please cite our original articles. 
The following Bibtex entry can be used:

@article{basar2020SimRIS,
Author = {E. {Basar} and I. {Yildirim}},
Booktitle = {Proc. IEEE Latin-American Conf. Commun. (LATINCOM 2020)},
Title= {{SimRIS} Channel Simulator for Reconfigurable Intelligent Surface-Empowered {mmWave} Communication Systems},
Year={2020},
month={Nov.},
Pages= {1-6},}

@article{basar2020SimRIS_2,
Author = {E. {Basar} and I. {Yildirim} and F. {Kilinc}},
journal = {IEEE Trans. Commun. (Early access)}
Title = {Indoor and Outdoor Physical Channel Modeling and Efficient Positioning for Reconfigurable Intelligent Surfaces in mm{W}ave Bands},
Year = {2021},
month={Sep.},}

@article{basar2021SimRIS_3,
Author = {E. {Basar} and I. {Yildirim}},
Journal= {IEEE Wireless Communications},
Title = {Reconfigurable Intelligent Surfaces for Future Wireless Networks: {A} Channel Modeling Perspective},
Year = {2021}, 
Volume={28},
Number={3},
Pages={108-114},
Doi= {10.1109/MWC.001.2000338},}