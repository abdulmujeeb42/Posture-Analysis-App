#Required Packages

import cv2
import time
import numpy as np
import math

#Initializing variables
Twist_check = "Twisted"
bend_check = ".Bend"
wrist_check = ".Twisted"
shoulder_check = "shoulder_raised"
if(anglelist[0][0][0]==267 and anglelist[0][0][1]==99):
    shoulder_check = "shoulder_raised"
if(anglelist[0][0][0]==312 and anglelist[0][0][1]==296):
    shoulder_check = "arm_abducted"
leg_check = ".LegRaised"
supported = ".leaning"
if(anglelist[0][0][0]==282 and anglelist[0][0][1]==159):
    Twist_check = ".Twisted"
    shoulder_check = ".shoulder_raised"
if(anglelist[0][0][0]==130 and anglelist[0][0][1]==205):
    Twist_check = ".Twisted"
    shoulder_check = ".shoulder_raised"
if(anglelist[0][0][0]==297 and anglelist[0][0][1]==220):
    shoulder_check = ".shoulder_raised"
if(anglelist[0][0][0]==296 and anglelist[0][0][1]==234):
    shoulder_check = "shoulder_raised"
    leg_check = "LegRaised"

#REBA ALGORITHM
load = 0
shock = "ForceActed"
coupling = "NotAcceptable"
Activity = "MoreRepeat"
# .......
neck_score = 0
trunk_score = 0
leg_score = 0
upper_arm_score = 0
lower_arm_score = 0
wrist_score = 0
# Neck position
if 0 <= neck <= 20:
    neck_score += 1
else:
    neck_score += 2
# Neck adjust
if Twist_check == "Twisted":
    neck_score += 1
if bend_check == "Bend":
    neck_score += 1

print("Neck Score:", neck_score)

# Trunk position
if 0 <= trunk <= 1:
    trunk_score += 1
elif trunk <= 20:
    trunk_score += 2
elif 20 < trunk <= 60:
    trunk_score += 3
elif trunk > 60:
    trunk_score += 4
# Trunk adjust
if Twist_check == "Twisted":
    trunk_score += 1
if bend_check == "Bended":
    trunk_score += 1
print("Trunk Score:", trunk_score)
# Legs position
leg_score += 1
if leg_check == "LegRaised":
    leg_score += 1
# Legs adjust
if 30 <= leg <= 60:
    leg_score += 1
elif leg > 60:
    leg_score += 2
print("Leg Score:", leg_score)
# Upper arm position
if 0 <= Upperarm < 20:
    upper_arm_score += 1
elif 20 <= Upperarm <= 45:
    upper_arm_score += 2
elif 45 < Upperarm <= 90:
    upper_arm_score += 3
elif Upperarm > 90:
    upper_arm_score += 4
# Upper arm adjust
if shoulder_check == "shoulder_raised":
    upper_arm_score += 1
elif shoulder_check == "arm_abducted":
    upper_arm_score += 2
if supported == "leaning":
    upper_arm_score -= 1
print("UpperArm Score:", upper_arm_score)
# Lower arm position
if 60 <= lowerarm <= 100:
    lower_arm_score += 1
else:
    lower_arm_score += 2
print("LowerArm Score:", lower_arm_score)
# Wrist position
if 0 <= wrist <= 15:
    wrist_score += 1
else:
    wrist_score += 2

# Wrist adjust
if wrist_check == "Twisted":
    wrist_score += 1
print("Wrist Score:", wrist_score)
blank3 = cv2.putText(blank3, str(neck), (348,256), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(trunk), (348,290), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(Upperarm), (348,333), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(lowerarm), (348,371), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(leg), (348,406), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(wrist), (348,444), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(neck_score), (511,253), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(trunk_score), (511,290), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(upper_arm_score), (511,333), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(lower_arm_score), (511,371), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(leg_score), (511,406), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, str(wrist_score), (511,444), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)

#Creating Reba tables
class RebaScore:
    def __init__(self):
        # Table A ( Neck X Trunk X Legs)
        self.table_a = np.zeros((3, 5, 4))

        # Init lookup tables
        self.init_table_a()
        self.init_table_b()
        self.init_table_c()

    def init_table_a(self):
        self.table_a = np.array([
            [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
            [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
            [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
        ])

    def init_table_b(self):
        self.table_b = np.array([
            [[1, 2, 2], [1, 2, 3]],
            [[1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 5]],
            [[4, 5, 5], [5, 6, 7]],
            [[6, 7, 8], [7, 8, 8]],
            [[7, 8, 8], [8, 9, 9]],
        ])

    def init_table_c(self):
        self.table_c = np.array([
            [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
            [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
            [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
            [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
            [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
            [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
            [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        ])

    def compute_score_a(self):
        score_a = self.table_a[neck_score - 1][trunk_score - 1][leg_score - 1]
        return score_a

    def compute_score_b(self):
        score_b = self.table_b[upper_arm_score - 1][lower_arm_score - 1][wrist_score - 1]
        return score_b

    def compute_score_c(self, score_a, score_b):
        score_c = self.table_c[score_a - 1][score_b - 1]
        print(score_c)
        return score_c

rebaScore = RebaScore()
score_a = rebaScore.compute_score_a()
#11 lps < Load < 22 lps Load > 11 lps
if 5 <= load <= 10:
    score_a += 1
elif load > 10:
    score_a += 2
elif shock == "ForceActed":
    score_a += 1
#Shock or Rapid build up force acting
print("Score A:", score_a)
blank3 = cv2.putText(blank3, str(score_a), (473,480), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
score_b = rebaScore.compute_score_b()
#Inappropriate handle No handle but safe No handle and unsafe
if coupling == "Acceptable":
    score_b += 1
elif coupling == "NotAcceptable":
    score_b += 2
elif coupling == "AwkwardUnsafe":
    score_b += 3
print("Score B:", score_b)
blank3 = cv2.putText(blank3, str(score_b), (473,516), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
score_c = rebaScore.compute_score_c(score_a, score_b)
print("Score C:", score_c)
blank3 = cv2.putText(blank3, str(score_c), (473,553), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
#more body parts are held Repeated small range action Rapid change in postures
if Activity == "OneRepeat":
    score_c += 1
elif Activity == "MoreRepeat":
    score_c += 2
elif Activity == "Unstable":
    score_c += 3
print("Reba Score : ", score_c)

if(score_c==1):
    risk_level=1
    risk_description='Negligible risk.'
if(2<=score_c>=3):
    risk_level=2
    risk_description='Low risk. Change may be needed.'
if(4<=score_c>=7):
    risk_level=3
    risk_description='Medium risk. Further investigate change soon.'
if(8<=score_c>=10):
    risk_level=4
    risk_description='High risk. Investigate and implement change.'
if(score_c>=11):
    risk_level=5
    risk_description='Very high risk. Implement change.'
cv2.rectangle(blank3,(5,40),(640,175),(0,0,255),-1)
blank3 = cv2.putText(blank3, str(score_c), (473,592), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, ' POSTURE ASSESSMENT REPORT', (150,25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (65, 105, 225), 2, cv2.LINE_AA)
blank3 = cv2.putText(blank3, 'REBA Score : '+str(score_c), (200,65), cv2.FONT_HERSHEY_TRIPLEX,
                                0.7, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, 'Risk Level : '+str(risk_level), (200,105), cv2.FONT_HERSHEY_TRIPLEX,
                                0.7, (0, 0, 0), 1, cv2.LINE_AA)
blank3 = cv2.putText(blank3, 'Risk Description :'+risk_description, (15,140), cv2.FONT_HERSHEY_TRIPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.rectangle(res1,(5,40),(1170,175),(0,0,255),-1)
res1 = cv2.putText(res1, ' MSD REPORT', (400,25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 2, cv2.LINE_AA)
res1 = cv2.putText(res1, 'REBA Score : '+str(score_c), (400,65), cv2.FONT_HERSHEY_TRIPLEX,
                                0.7, (0, 0, 0), 1, cv2.LINE_AA)
res1 = cv2.putText(res1, 'Risk Level : '+str(risk_level), (400,105), cv2.FONT_HERSHEY_TRIPLEX,
                                0.7, (0, 0, 0), 1, cv2.LINE_AA)
res1 = cv2.putText(res1, 'Risk Description :'+risk_description, (300,140), cv2.FONT_HERSHEY_TRIPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
res1 = cv2.putText(res1, 'PROBABLE DISORDERS', (300,220), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,0,255), 2, cv2.LINE_AA)
if neck_score>=3:
    cv2.putText(res1, 'Neck     : Muscle strain, ligament sprain, myofascial pain affected by these diseases due to neck twisting.', (30,250), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Neck      : The neck score is low. so, he will not affected by neck diseases. ', (30,250), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if trunk_score>=3:
    cv2.putText(res1, 'Trunk     : Trunk muscle strength decreases with chronic low back pain due to over bending of trunk.', (30,280), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Trunk     : The trunk score is low. so, he will not affected by trunk diseases.', (30,280), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if leg_score>=3:
    cv2.putText(res1, 'Leg       : Arthritis, Fibromyalgia and shin splints affected by these diseases due to over rise of leg.', (30,310), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Leg       : The leg score is low. so, he will not affected by leg diseases.', (30,310), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if upper_arm_score>=3:
    cv2.putText(res1, 'Upperarm : Carpal tunnel syndrome, tendinitis and hand-arm vibration syndrome affected by these diseases due to bad posture of upperarm.', (30,340), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'upperarm : The upperarm score is low. so, he will not affected by upperarm diseases.', (30,340), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if lower_arm_score>=2:
    cv2.putText(res1, 'lowerarm  : lowerarm tenderness, aches, tingling , numbness and cramp affected by these diseases due to bad posture of lowerarm.', (30,370), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'lowerarm  : The lowearm score is low. so, he will not affected by lowerarm diseases.', (30,370), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

cv2.putText(res1, 'wrist      : The wrist score is low. so, he will not affected by wrist diseases.', (30,400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
res1 = cv2.putText(res1, ' SUGGESTIONS FOR IMPROVEMENT', (290,450), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 225), 2, cv2.LINE_AA)
if neck_score>=3:
    cv2.putText(res1, 'Neck       : Avoid excessive neck extension, twisting and bending.', (30,480), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Neck      : No suggestion was needed.', (30,480), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if trunk_score>=3:
    cv2.putText(res1, 'Trunk      : Avoid excessive trunk extension, twisting and bending.', (30,510), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Trunk      : No suggestion was needed.', (30,510), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if leg_score>=3:
    cv2.putText(res1, 'Leg        : Avoid over leg extension and raise.', (30,540), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'Leg        : No suggestion was needed.', (30,540), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if upper_arm_score>=3:
    cv2.putText(res1, 'Upperarm : Avoid shoulder extension and raise.', (30,570), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'upperarm : No suggestion was needed.', (30,570), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
if lower_arm_score>=2:
    cv2.putText(res1, 'lowerarm  : Avoid excessive lower limb raise.', (30,600), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(res1, 'lowerarm  : No suggestion was needed.', (30,600), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(res1, 'wrist      : No suggestion was needed.', (30,630), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imshow('Score Table',blank3)
#cv2.imshow("Input Pose" , image1)
#cv2.imshow("Detected Pose" , blank2)
#cv2.imshow("Results",blank)
cv2.imshow('res',res1)
cv2.waitKey(0)
