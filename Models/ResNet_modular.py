import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        """
        
        Struktura ResNet mreže.

        1 konvolucijska plast (k=7, s=2, p=3).
        4 plasti, sestavljene iz residualnih blokov, definiranih v zgornjem classu. Št. blokov v plasti se definira ob klicu classa.
        Nakoncu avgpool (se ne rabi) in fc plast 512->10 (št. razredov v CIFAR10).

        Vhodi:
            Tip bloka: Residualni blok
            Arrray plasti/blokov: array, ki določi, koliko blokov sestavlja posamezno plast
            Št. razredov: 10
        
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3), # ven pride 16 x 16 x 64
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)   # ven pride 8 x 8 x 64
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)    # ohrani se dimenzija 8 x 8 x 64, se pa izvaja konvolucija
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)   # pride ven 4 x 4 x 128
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)   # pride ven 2 x 2 x 256
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)   # pride ven 1 x 1 x 512
        #self.avgpool = nn.AvgPool2d(7, stride=1)    #ne rabi bit kernel 7, ker je itak slika že 1x1
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        
        Sestavlja sekvence (residualnih) blokov v plasti mreže.
        Na podlagi velikosti maske, koraka (stride) in dodajanja (padding) plast spreminja ali ohranja dimenzijo obdelane slike.
        Tudi v primeru, da se dimenzije ne zmanjšajo in se samo izvaja konvolucija, se mreža uči.
        Blokom ni potrebno, da zmanjšujejo dimenzijo ali spremeniti število filtrov, takrat imajo stride/korak = 1 in so brez downsampla.

        Vhodi:
            block: Tip bloka(residual block)
            planes: št. filtrov
            blocks: število blokov v posamezni plasti
            stride: velikost koraka (v prvi konvolucijski plasti)
        
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:  # downsample se nastavi, če se zmanjšujejo dimenzije slike/št. vhodnih kanalov se ne ujema s št. izhodnih
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [] # prazen array za držanje residualnih blokov
        layers.append(block(self.inplanes, planes, stride, downsample))  # Residualni blok se ustvari in doda k layers
        self.inplanes = planes  # št. izhodnih filtrov se shrani v spremenljivko št. vhodnih filtrov

        for i in range(1, blocks):  # ustvarimo toliko layerjev, kolikor je elementov v arrayu, s katerim kličemo model
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)     

        # * je unpacking operator, odpakira elemente v listu layers, da so lahko podani kot individualni argumenti v nn.Sequential konstruktor
        # V pythonu se * uporablja pred listom/tuplom pri klicu funkcije, ko odpakira iterable in poda elemente kot ločene argumente

    def forward(self, x):   # skip connection
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        """

        Zgradba Residual bloka.

        Prva konvolucijska plast zmanjšuje dimenzijo glede na stride,
        druga plast izvaja konvolucijo, a ohrani dimenzijo slike.

        Vhodi:
            št. vhodnih kanalov
            št. izhodnih kanalov
            stride - korak
            downsample - skip connection

        Prva konvolucijska plast dejansko lahko spremeni dimenzijo obdelane slike, če je pri klicu funkcije stride > 1 (zmanjša dimenzijo slike).
        Če je stride = 1, se pri teh parametrih dimenzija slike ohrani in se izvaja samo konvolucija.
        Druga plast ima default vrednost stride = 1, torej ne spreminja velikosti slike.
        
        """
        self.conv1 = nn.Sequential(     
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),  
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels)
                    )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):   # skip connection
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample: # če je 1, gre delat skp connection
            residual = self.downsample(x)
        out += residual # doda se skip vrednost
        out = self.relu(out)
        return out

