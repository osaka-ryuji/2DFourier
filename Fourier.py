import numpy
import numpy.fft.fftpack as fftpack
from PIL import Image
import matplotlib.pyplot as plt

maxWidth = 256

class FFT:
#画像を読み込み、グレースケールにして、画像のサイズを測り規定値より大きかったらリサイズする処理。(軽量化のため)    
    def __init__(self):
        self.fig = plt.figure()
        
        originalImage = Image.open('F.jpg')
        (ow,oh) = originalImage.size
        
        if ow>maxWidth :
            monoImageArray = numpy.asarray(originalImage.convert('L').resize((maxWidth,oh*maxWidth//ow)))
        else:
            monoImageArray = numpy.asarray(originalImage.convert('L'))
#ここまで

# 画像をフーリエ変換し、その平均と標準偏差を求める処理。        
        (self.imageHeight, self.imageWidth) = monoImageArray.shape
        self.samples = numpy.zeros((self.imageHeight, self.imageWidth), dtype=complex)
        self.samplePoints = numpy.zeros((self.imageHeight, self.imageWidth, 4))
        self.fftImage = fftpack.fft2(monoImageArray)
        self.fftImageForPlot = numpy.roll(numpy.roll(numpy.real(self.fftImage), self.imageHeight//2, axis=0), self.imageWidth//2, axis=1)
        self.fftMean = numpy.mean(self.fftImageForPlot)
        self.fftStd = numpy.std(self.fftImageForPlot)
#ここまで

#元の画像を表示する処理。        
        self.axes1 = plt.subplot(2,3,1)
        plt.imshow(monoImageArray, cmap='gray')
#ここまで

#2次元フーリエ変換した画像を表示する処理。      
        self.axes2 = plt.subplot(2,3,2)
        p = plt.imshow(self.fftImageForPlot, cmap='gray')
        p.set_clim(self.fftMean-self.fftStd, self.fftMean+self.fftStd)
#ここまで
        
#画像と同じサイズの枠を表示する処理３つ。         
        self.axes3 = plt.subplot(2,3,4)
        self.axes3.set_aspect('equal')
        self.axes3.set_xlim(0,self.imageWidth)
        self.axes3.set_ylim(self.imageHeight,0)

        self.axes4 = plt.subplot(2,3,5)
        self.axes4.set_aspect('equal')
        self.axes4.set_xlim(0,self.imageWidth)
        self.axes4.set_ylim(self.imageHeight,0)
        
        self.axes5 = plt.subplot(2,3,6)
        self.axes5.set_aspect('equal')
        self.axes5.set_xlim(0,self.imageWidth)
        self.axes5.set_ylim(self.imageHeight,0)
#ここまで

#マウス操作の読み込み処理。       
        self.bMousePressed = False
        self.mouseButton = 0
        self.bCtrlPressed = False
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.fig.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.fig.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.fig.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        
        plt.show()
#ここまで

#クリックされているときの処理     
    def onButtonPress(self, event):
        self.bMousePressed = True
        self.mouseButton = event.button
        self.update(event)
#ここまで

#クリックが解除したときの処理     
    def onButtonRelease(self, event):
        self.bMousePressed = False
        self.mouseButton = 0
#ここまで

#マウスが動いているときの処理    
    def onMove(self, event):
        if self.bMousePressed:
            self.update(event)
#ここまで

#controlキーが押下されているときの処理    
    def onKeyPress(self, event):
        if event.key == 'control':
            self.bCtrlPressed = True
#ここまで

#controlキーが押下されていないときの処理    
    def onKeyRelease(self, event):
        if event.key == 'control':
            self.bCtrlPressed = False
#ここまで

#クリックした場所に対応するフーリエ変換のパラメータ表示とそれを重ね合わせた画像を表示する処理。
    def update(self, event):
        if event.inaxes != self.axes4:
            return
        
        if event.xdata != None:
            x = (int(event.xdata)+self.imageWidth//2)%self.imageWidth
            y = (int(event.ydata)+self.imageHeight//2)%self.imageHeight
            
            plt.sca(self.axes5)
            plt.cla()
            waveImg = numpy.zeros((self.imageHeight,self.imageWidth))
            waveImg[y,x] = 1
            plt.imshow(numpy.real(fftpack.ifft2(waveImg)), cmap='gray')
            
            if not self.bCtrlPressed:
                bNeedUpdate = False
                if self.samples[y,x] != self.fftImage[y,x] and self.mouseButton == 1: #left button
                   bNeedUpdate = True
                   self.samples[y,x] = self.fftImage[y,x]
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,0] = 1
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,3] = 1
                elif self.samples[y,x] != numpy.complex(0.0,0.0) and self.mouseButton == 3: #right button
                   bNeedUpdate = True
                   self.samples[y,x] = numpy.complex(0.0,0.0)
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,0] = 0
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,3] = 0
                
                if bNeedUpdate:
                    plt.sca(self.axes4)
                    plt.cla()
                    p = plt.imshow(self.fftImageForPlot, cmap='gray')
                    p.set_clim(self.fftMean-self.fftStd,self.fftMean+self.fftStd)
                    plt.imshow(self.samplePoints)
                    
                    plt.sca(self.axes3)
                    plt.cla()
                    plt.imshow(numpy.real(fftpack.ifft2(self.samples)), cmap='gray')
                    
            else:
                for xi in range(x-self.imageWidth//32, x+self.imageWidth//32):
                    for yi in range(y-self.imageWidth//32, y+self.imageWidth//32):
                        if xi>=self.imageWidth:
                            xx = xi-self.imageWidth
                        else:
                            xx = xi
                        if yi>=self.imageHeight:
                            yy = yi-self.imageHeight
                        else:
                            yy = yi
                        if self.mouseButton == 1: #left button
                            self.samples[yy,xx] = self.fftImage[yy,xx]
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,0] = 1
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,3] = 0.7
                        elif self.mouseButton == 3: #right button
                            self.samples[yy,xx] = numpy.complex(0.0,0.0)
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,0] = 0
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,3] = 0
                plt.sca(self.axes4)
                plt.cla()
                plt.imshow(self.samplePoints)
                
                plt.sca(self.axes3)
                plt.cla()
                plt.imshow(numpy.real(fftpack.ifft2(self.samples)), cmap='gray')
            
            self.fig.canvas.draw()

if __name__ == '__main__':
    FFT()
