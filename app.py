import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import cv2
import imutils
import numpy as np
import tensorflow
from imutils.perspective import four_point_transform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
import sys
from copy import deepcopy

try:
    model = tensorflow.keras.models.open('model.h5')
except Exception:
    st.write("Error loading predictive model")
    
def output(a):
    sys.stdout.write(str(a))


N = 9

def read(field):
    """ Read field into state (replace 0 with set of possible values) """

    state = deepcopy(field)
    for i in range(N):
        for j in range(N):
            cell = state[i][j]
            if cell == 0:
                state[i][j] = set(range(1,10))

    return state




def done(state):
    """ Are we done? """

    for row in state:
        for cell in row:
            if isinstance(cell, set):
                return False
    return True


def propagate_step(state):
    """
    Propagate one step.

    @return:  A two-tuple that says whether the configuration
              is solvable and whether the propagation changed
              the state.
    """

    new_units = False

    # propagate row rule
    for i in range(N):
        row = state[i]
        values = set([x for x in row if not isinstance(x, set)])
        for j in range(N):
            if isinstance(state[i][j], set):
                state[i][j] -= values
                if len(state[i][j]) == 1:
                    val = state[i][j].pop()
                    state[i][j] = val
                    values.add(val)
                    new_units = True
                elif len(state[i][j]) == 0:
                    return False, None

    # propagate column rule
    for j in range(N):
        column = [state[x][j] for x in range(N)]
        values = set([x for x in column if not isinstance(x, set)])
        for i in range(N):
            if isinstance(state[i][j], set):
                state[i][j] -= values
                if len(state[i][j]) == 1:
                    val = state[i][j].pop()
                    state[i][j] = val
                    values.add(val)
                    new_units = True
                elif len(state[i][j]) == 0:
                    return False, None

    # propagate cell rule
    for x in range(3):
        for y in range(3):
            values = set()
            for i in range(3 * x, 3 * x + 3):
                for j in range(3 * y, 3 * y + 3):
                    cell = state[i][j]
                    if not isinstance(cell, set):
                        values.add(cell)
            for i in range(3 * x, 3 * x + 3):
                for j in range(3 * y, 3 * y + 3):
                    if isinstance(state[i][j], set):
                        state[i][j] -= values
                        if len(state[i][j]) == 1:
                            val = state[i][j].pop()
                            state[i][j] = val
                            values.add(val)
                            new_units = True
                        elif len(state[i][j]) == 0:
                            return False, None

    return True, new_units

def propagate(state):
    """ Propagate until we reach a fixpoint """
    while True:
        solvable, new_unit = propagate_step(state)
        if not solvable:
            return False
        if not new_unit:
            return True


def solve(state):
    """ Solve sudoku """

    solvable = propagate(state)

    if not solvable:
        return None

    if done(state):
        return state

    for i in range(N):
        for j in range(N):
            cell = state[i][j]
            if isinstance(cell, set):
                for value in cell:
                    new_state = deepcopy(state)
                    new_state[i][j] = value
                    solved = solve(new_state)
                    if solved is not None:
                        return solved
                return None
    
    
def detect(im):
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7, 7), 3)
  
  thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  thresh = cv2.bitwise_not(thresh)
  
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  
  puzzleCnt = None
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      puzzleCnt = approx
      break
      
  if puzzleCnt is None:
    return -1
  else:
    output = im.copy()
    cv2.drawContours(output, [puzzleCnt],-1,(0, 255, 0), 2)
   
 
  puzzle = four_point_transform(im, puzzleCnt.reshape(4, 2))
  warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

  stepX = warped.shape[1] // 9
  stepY = warped.shape[0] // 9

  loc = []

  for i in range(0,9):
    tmp = []
    for j in range(0,9):
      x = (j*stepX,i*stepY,(j+1)*stepX,(i+1)*stepY)
      tmp.append(x)
    loc.append(tmp)
   
  grid = []
  printgrid = []

  for i in range(9):
    tmp = []
    for j in range(9):
      cell = warped[loc[i][j][1]:loc[i][j][3],loc[i][j][0]:loc[i][j][2]]
      thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      thresh  = clear_border(thresh)
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      
      if len(cnts) == 0:
        tmp.append(0)
        printgrid.append(loc[i][j])
      else:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)
      
      if percentFilled < 0.03:
        tmp.append(0)
        printgrid.append(loc[i][j])
      else:
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)
        roi = cv2.resize(digit,(28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        tmp.append(model.predict(roi).argmax())
   
   
    grid.append(tmp)
      

  ans = []
  for i in range(9):
    for j in range(9):
      if grid[i][j] == 0:
        ans.append((i,j))
  

  field = grid

  state = read(field)
         
  grid = solve(state)

  puz = puzzle.copy()


  k = 0
  for i in printgrid:
    textX = int((i[2] - i[0]) * 0.33)
    textY = int((i[3] - i[1]) * -0.2)
    textX += i[0]
    textY += i[3]
    x = ans[k][0]
    y = ans[k][1]
    cv2.putText(puz, str(grid[x][y]), (textX, textY),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
    k += 1
  
  return puz
  
 
 
def main():
    st.title("Sudoku Solver.ai ")
    st.write("**Using the Convolutional Neural Network**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
    	uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if uploaded_file is not None:

                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

                image = cv2.imdecode(file_bytes, 1)

                if st.button("Process"):
                   result_img = detect(image=image)
                   st.image(result_img, use_column_width = True)

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
