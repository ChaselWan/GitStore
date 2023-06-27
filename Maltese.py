# DeskTop Pet Maltese
import PyQt5

class Maltese(object):
  def __init__(self, parent=None, **kwargs):
    super(Maltese, self).__init__(parent)
    self.show()
    # 一些参数
    self.ROOT_DIR = 'E:\Users\Admin\Desktop\日程表'
    self.pet_name = 'Maltese'
    self.actions = {'hug':[str(i) for i in range(1, 6)]}
    
    # 初始化
    self.setWindowFlags(Qt.FramelessWindowHint|Qt.WindowStaysOnTopHint|Qt.SubWindow)
    self.setAutoFillBackground(False)
    self.setAttribute(Qt.WA_TranslucentBackground, True)
    self.repaint()
    #导入一个线条小狗
    self.pet_images, iconpath = self.LoadPetImages()
    # 设置退出选项
    quit_action = QAction('退出', self, triggered=self.quit)
    quit_action.setIcon(QIcon(iconpath))
    self.tray_icon_menu = QMenu(self)
    self.tray_icon_menu.addAction(quit_action)
    self.tray_icon = QSystemTrayIcon(self)
    self.tray_icon.setIcon(QIcon(iconpath))
    self.tray_icon.setContextMenu(self.tray_icon_menu)
    self.tray_icon.show()

    # 当前显示的图片
    self.image = QLabel(self)
    self.setImage(self.pet_images[0][0])
    # 是否跟随鼠标
    self.is_follow_mouse = False
    # 宠物拖拽时避免鼠标直接跳到左上角
    self.mouse_drag_pos = self.pos()
    # 显示
    self.resize(self.pet_images[0][0].size().width(), self.pet_images[0][0].size().height())
    self.randomPosition()
    self.show()
    # 宠物动画动作执行所需的一些变量
    self.is_running_action = False
    self.action_images = []
    self.action_pointer = 0
    self.action_max_len = 0
    # 每隔一段时间做个动作
    self.timer_act = QTimer()
    self.timer_act.timeout.connect(self.randomAct)
    self.timer_act.start(500)

  def LoadPetImages(self):
    # 与源代码不同，我需要将一个线条小狗存成 Maltese/动作场景/item.png
    # 所以actions应该是一个dict
    pet_images_dict = {}
    pet_images = []
    for action in self.actions.keys: # action = 'hug' or else
      pet_images.append([self.loadImage(os.path.join(self.ROOT_DIR, self.pet_name, action, item + '.png')) for item in action])  # pet_images = [[],[],[],[],...]
    iconpath = os.path.join(self.ROOT_DIR, pet_name, action, '1.png')  # 最初出现的图像的地址，轻轻地功德小狗一下
    return pet_images_dict, iconpath

  '''鼠标左键按下时, 宠物将和鼠标位置绑定'''
  def mousePressEvent(self, event):
    if event.button() == Qt.LeftButton:
      self.is_follow_mouse = True
      self.mouse_drag_pos = event.globalPos() - self.pos()
      event.accept()
      self.setCursor(QCursor(Qt.OpenHandCursor))
  '''鼠标移动, 则宠物也移动'''
  def mouseMoveEvent(self, event):
    if Qt.LeftButton and self.is_follow_mouse:
      self.move(event.globalPos() - self.mouse_drag_pos)
      event.accept()
  '''鼠标释放时, 取消绑定'''
  def mouseReleaseEvent(self, event):
    self.is_follow_mouse = False
    self.setCursor(QCursor(Qt.ArrowCursor))

  '''导入图像'''
  def loadImage(self, imagepath):
    image = QImage()
    image.load(imagepath)
    return image

  '''随机到一个屏幕上的某个位置'''
  def randomPosition(self):
    screen_geo = QDesktopWidget().screenGeometry()
    pet_geo = self.geometry()
    width = (screen_geo.width() - pet_geo.width()) * random.random()
    height = (screen_geo.height() - pet_geo.height()) * random.random()
    self.move(width, height)

  '''退出程序'''
  def quit(self):
    self.close()
    sys.exit()
    
    
  '''随机做一个动作'''
  def randomAct(self):
    if not self.is_running_action:  # not False
      self.is_running_action = True
      self.action_choice = random.choice(self.actions.keys)  
      self.action_images = self.pet_images_dict[self.action_choice]  # 把一个动作的所有图像赋值
      self.action_max_len = len(self.action_images)
      self.action_pointer = 0
    self.runFrame()

    '''完成动作的每一帧'''
  def runFrame(self):
    while is_running_action:
      if self.action_pointer == self.action_max_len:
        self.is_running_action = False
        self.action_pointer = 0
        self.action_max_len = 0

      self.setImage(self.action_images[self.action_pointer])
      self.action_pointer += 1
    
    '''设置当前显示的图片'''
  def setImage(self, image):
    self.image.setPixmap(QPixmap.fromImage(image))
    
