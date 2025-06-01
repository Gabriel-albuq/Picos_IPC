import gxipy as gx
import sys
import cv2
import numpy as np
import time

class GxiCapture:
    def __init__(self, index=0):
        self.device_manager = gx.DeviceManager()
        self.dev_num, self.dev_info_list = self.device_manager.update_all_device_list()
        self._is_opened = False
        self._height = 0
        self._width = 0
        self._fps = 30  # Valor padrão, será atualizado
        
        if self.dev_num == 0:
            print("Nenhuma câmera encontrada.")
            sys.exit(1)
        
        try:
            self.cam = self.device_manager.open_device_by_index(index)
            self._is_opened = True
            
            # Configurações iniciais
            self.cam.TriggerMode.set(1)
            self.cam.ExposureTime.set(8000.0)
            self.cam.GainAuto.set(1)
            self.cam.DigitalShift.set(1)
            self.cam.SensorShutterMode.set(0)
            self.cam.BalanceWhiteAuto.set(1)

            # Melhorias na imagem
            self.gamma_lut = gx.Utility.get_gamma_lut(self.cam.GammaParam.get()) if self.cam.GammaParam.is_readable() else None
            self.contrast_lut = gx.Utility.get_contrast_lut(self.cam.ContrastParam.get()) if self.cam.ContrastParam.is_readable() else None
            self.color_correction_param = self.cam.ColorCorrectionParam.get() if self.cam.ColorCorrectionParam.is_readable() else 0
            
            self.cam.stream_on()
            
            # Obter propriedades iniciais
            try:
                self._fps = self.cam.AcquisitionFrameRate.get() if self.cam.AcquisitionFrameRate.is_readable() else 30
            except:
                self._fps = 30

            try:
                self._height = self.cam.Height.get() if self.cam.Height.is_readable else 0
                self._width = self.cam.Width.get() if self.cam.Width.is_readable else 0
            except:
                self._height  = 0
                self._width = 0
            
        except Exception as e:
            print(f"Erro ao abrir a câmera: {e}")
            self._is_opened = False

    def isOpened(self):
        """Verifica se a câmera está aberta e funcionando"""
        return self._is_opened

    def get(self, prop_id):
        """
        Obtém propriedades da câmera similar ao OpenCV
        """
        if not self.isOpened():
            return 0.0
        
        try:
            if prop_id == cv2.CAP_PROP_FPS:
                return float(self._fps)
                
            elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self._width)
                
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self._height)
                
            elif prop_id == cv2.CAP_PROP_EXPOSURE:
                try:
                    return self.cam.ExposureTime.get() / 1000  # Convertendo para ms
                except:
                    return -1.0  # Retorna -1 se não for suportado
                    
            else:
                return 0.0  # Retorna 0 para propriedades não implementadas
                
        except Exception as e:
            print(f"Erro ao obter propriedade {prop_id}: {e}")
            return 0.0

    def read(self):
        if not self.isOpened():
            return False, np.zeros((self._width, self._height, 3), dtype=np.uint8)
            
        try:
            self.cam.TriggerSoftware.send_command()
            raw_image = self.cam.data_stream[0].get_image()
            
            if raw_image is None:
                return False, np.zeros((self._width, self._height, 3), dtype=np.uint8)

            rgb_image = raw_image.convert("RGB")
            if rgb_image is None:
                return False, np.zeros((self._width, self._height, 3), dtype=np.uint8)

            try:
                rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
                numpy_image = rgb_image.get_numpy_array()
                numpy_image_bgr = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                
                return True, numpy_image_bgr
                
            except Exception as e:
                print(f"Erro no processamento da imagem: {e}")
                return False, np.zeros((self._width, self._height, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Erro durante a captura: {e}")
            self._is_opened = False
            return False, np.zeros((self._width, self._height, 3), dtype=np.uint8)

    def release(self):
        if hasattr(self, 'cam') and self.cam:
            try:
                self.cam.stream_off()
                self.cam.close_device()
                self._is_opened = False
            except Exception as e:
                print(f"Erro ao liberar a câmera: {e}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = GxiCapture(0)
    
    if not cap.isOpened():
        print("Não foi possível abrir a câmera")
        sys.exit(1)
    
    # Obtendo propriedades como no OpenCV
    device_fps = cap.get(cv2.CAP_PROP_FPS)
    device_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    device_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    device_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    print(f"FPS: {device_fps}")
    print(f"Resolução: {device_width}x{device_height}")
    print(f"Exposição: {device_exposure} ms")
    
    while True:
        time.sleep(0.01)
        ret, frame = cap.read()
        
        if ret:
            cv2.imshow("Imagem Capturada", frame)
        else:
            print("Falha ao capturar frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()