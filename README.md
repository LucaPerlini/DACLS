Il preprocessing segue questi passaggi:

1- Viene scaricato il dataset (SingFake) tramite lo script CreateDataset3.py;
2- Viene applicato il Demucs.py al dataset;
3- Successivamente viene applicato il VAD con VADPyAnnote.py;
4- Infine, si applica SingleChannel.py che seleziona randomicamente se mantenere il canale destro o sinistro della traccia, eliminando l'altro.
5- LogSpectrogramV2.py taglia le tracce in durata da 1,28 secondi con frequenza di campionamento 16 kHz e genera spettrogrammi 128x128 con hop-size 10 ms e frame-size 20 ms.
