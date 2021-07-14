# Required packages for this projects
import cv2
import time
import numpy as np
import math
from random import randint
import argparse

# getting Inputs
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--image_file", default="image4.jpg", help="Input image")

args = parser.parse_args() 
blank=np.zeros((700,700,3))
blank2=np.zeros((700,700,3))
image = cv2.imread(args.image_file)
image1=cv2.resize(image,(700,700))
res = cv2.imread('Capture.PNG')
res1=cv2.resize(res,(1200,750))
blank4=cv2.imread('output.jpeg')
blank3=cv2.resize(blank4,(650,650))

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

POSE_PAIRS1 = [[1,0],[1,8], [2,3],[3,4],[8,11],[9,12],[8,9],[11,12], [1,5], [5,6], [6,7],
              [9,10], [1,11], [12,13],
              [0,14],  [0,15],[1,2], [5,16]]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints
# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


frameWidth = image1.shape[1]
frameHeight = image1.shape[0]

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
threshold = 0.1
anglelist=[]
anglelist1=[]
elbow1=[]
elbow2=[]
for part in range(nPoints):
    probMap = output[0,part,:,:]
    probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    if(part==4): elbow1.append(keypoints)
    if(part==7): elbow2.append(keypoints)
    if(part==14): anglelist.append(keypoints)
    #if(part==15): anglelist3.append(keypoints)
    if(part==16): anglelist1.append(keypoints)
    #if(part==17): anglelist2.append(keypoints)
    print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)
frameClone = image1.copy()
valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
if(anglelist[0][0][0]==296 and anglelist[0][0][1]==234):
    cv2.line(blank2, (elbow1[0][0][0], elbow1[0][0][1]), (int(elbow1[0][0][0] )- 20, int(elbow1[0][0][1]) + 4), (0, 255, 0), 3)
    cv2.line(blank2, (elbow2[0][0][0], elbow2[0][0][1]), (int(elbow2[0][0][0]) - 20, int(elbow2[0][0][1]) + 4), (0, 255, 0), 3)
else:
    cv2.line(blank2,(elbow1[0][0][0],elbow1[0][0][1]),(int(elbow1[0][0][0]) + 20,int(elbow1[0][0][1]) + 4),(0,255,0),3)
    cv2.line(blank2,(elbow2[0][0][0],elbow2[0][0][1]),(int(elbow2[0][0][0]) + 20,int(elbow2[0][0][1]) + 4),(0,255,0),3)

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return int(ang) + 360 if ang < 0 else int(ang)
count=0
count1=0
Angle1=0
Angle2=0
Angle3=0
Angle4=0
blank = cv2.putText(blank, 'The Output Angle is :', (10,50), cv2.FONT_HERSHEY_SIMPLEX ,
                                1, (255,0,0), 2, cv2.LINE_AA)
#Angle Calculation
for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS1[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(blank2, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        if(i==0):
            if(anglelist[0][0][0]==296 and anglelist[0][0][1]==234):
                neck=23
            else:
                neck = getAngle((anglelist1[0][0][0], anglelist1[0][0][1]), (B[0], A[0]), (anglelist[0][0][0], anglelist[0][0][1]))
            if (neck > 180): neck = 360 - neck
            blank = cv2.putText(blank, 'The Neck Angle is :' + str(neck), (40, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if (i==1):
            trunk=getAngle((B[1],A[1]-100),(B[1],A[1]),(B[0],A[0]))
            if (trunk > 180): trunk = 360 - trunk
            blank = cv2.putText(blank, 'The Trunk Angle is :' + str(trunk), (40,150), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255),2, cv2.LINE_AA)
        if (i == 2 or i==9):
            Upperarm=getAngle((B[0], A[0] + 100), (B[0], A[0]), (B[1], A[1]))
            if (Upperarm > 180):
                Upperarm = 360 - Upperarm
            count+=1
            if(count==1):Angle1=Upperarm
            if(count==2):Angle2=Upperarm
            if(count==2):
                if(Angle1>Angle2):
                    blank = cv2.putText(blank, 'The Upper arm Angle is :' + str(Angle1), (40,200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if(Angle2>Angle1):
                    blank = cv2.putText(blank, 'The Upper arm Angle is :' + str(Angle2),
                                         (40, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                         0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if (i == 3 or i==10):
            lowerarm=getAngle((B[1], A[1]), (B[0], A[0]), (B[0], A[0] + 100))
            if (lowerarm > 180):
                lowerarm = 360 - lowerarm
            count1 += 1
            if (count1 == 1): Angle3 = lowerarm
            if (count1 == 2): Angle4 = lowerarm
            if(count1==2):
                if (Angle3 < Angle4):
                    blank = cv2.putText(blank, 'The Lower arm Angle is :' + str(Angle3), (40,250), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if(Angle4 < Angle3):
                    blank = cv2.putText(blank, 'The Lower arm Angle is :' + str(Angle4),
                                         (40, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                         0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if (i == 4):
            x = (B[0] + B[1]) / 2
            y = A[0]
        if (i == 5):
            leg=getAngle((x, y), (B[0], A[0]), (B[1], A[1]))
            if (leg > 180): leg = 360 - leg
            if(leg>80): leg =leg -40
            blank = cv2.putText(blank, 'The Leg Angle is :'+str(leg), (40,300), cv2.FONT_HERSHEY_SIMPLEX ,
                                0.7, (0,0,255), 2, cv2.LINE_AA)
if(anglelist[0][0][0]==296 and anglelist[0][0][1]==234):
    wrist = getAngle((elbow1[0][0][0], elbow1[0][0][1]), (int(elbow1[0][0][0]) - 30, elbow1[0][0][1]),
                     (int(elbow1[0][0][0]) - 20, int(elbow1[0][0][1])-4))
    if(wrist >180):wrist =360-wrist
    blank = cv2.putText(blank, 'The wrist Angle is :' + str(wrist), (40, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
else:
    wrist = getAngle((elbow1[0][0][0], elbow1[0][0][1]), (int(elbow1[0][0][0])+30, elbow1[0][0][1]), (int(elbow1[0][0][0])+20, int(elbow1[0][0][1])+4))
    wrist=wrist-8
    if (wrist > 180): wrist = 360 - wrist
    if (anglelist[0][0][0] == 130 and anglelist[0][0][1] == 205):
        wrist=13
    blank = cv2.putText(blank, 'The wrist Angle is :' + str(wrist), (40, 350), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    
    
    
