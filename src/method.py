import numpy as np
import pandas as pd
from src.Eval import i_s

class idea(i_s):
    def __init__(self,rand=True):
        # To ceate Random in Selection
        self.rand=rand

    def get_param(self):
        pass

    def __choise_enemy(self, X, class_, frac=0.3):
        selected = []
        #for each class of data
        for value in np.unique(X[:, -1]):
            #if class same as selected class skip it
            if value == class_:
                continue

            # get index of elements with this class
            indices = np.where(X[:, -1] == value)[0]

            #randomly select index with frac size
            selected_index = np.random.choice(indices, size=int(len(indices) * frac))

            # get the value at the selected index
            selected_value = X[selected_index]
            selected.extend(selected_value)

        return selected

    def get_idx(self):
        return self.final_idx
    def eval_(self, model, X_test,X_train=None,k=5):
        if X_train is None:
            X_train = self.final_data[:, :]
            # y_train = self.final_data[:, -1]

        score,f1 = self.eval(model, X_train[:,:-1], X_train[:,-1], X_test[:,:-1], X_test[:,-1],k)
        self.f1=f1*100
        self.score=score*100


    def get_f1(self):
        return self.f1
    def get_score(self):
        return self.score
    def get_idx(self):
        return self.final_idx

    def get_data(self):
        return self.final_data

    def __method1(self, sc_li):

        idx = []
        # in each bin except last on get probability
        for j, i in enumerate(self.bins[1:-1]):
            frac = self.p_hits[j]
            #skip if no fraction
            if frac == 0:
                continue

            #convert score list to dataframe and get data in that range
            df = pd.DataFrame(sc_li)
            df = df[df.iloc[:, 1] < i]
            df = df[df.iloc[:, 1] >= i - (self.bins[1] - self.bins[0])]

            #apply fraction on list score and save its index
            val_idx = df.sample(frac=frac)[0].values
            idx.extend(val_idx.astype('int'))

        return idx

    def __method2(self, sc_li):
        idx = []

        # in each bin except last on calculate fraction(1-0.125 => 87% Fraction)
        for i in self.bins[1:-1]:
            if i <0.625:
                frac = 1 - i

            #convert score list to dataframe and get data in that range
            df = pd.DataFrame(sc_li)
            df = df[df.iloc[:, 1] < i]
            df = df[df.iloc[:, 1] >= i - (self.bins[1] - self.bins[0])]

            #apply fraction on list score and save its index
            val_idx = df.sample(frac=frac)[0].values
            idx.extend(val_idx.astype('int'))

        return idx

    def get_data(self):
        return self.final_data

    def i_s(self, data, K_neighbors=5, frac=0.3):

        #insert data index in first column of array
        # data = np.array(np.insert(data, 0, range(len(data)), axis=1))

        # Get number of class in dataset
        unique_class = np.unique(data[:, -1].astype('int'))
        li_idx = []
        sc_li = []

        for i in unique_class:
            pre_prune_data = []

            same_class_data = data[data[:, -1] == i]  ##select current class data

            pre_prune_data.extend(self.__choise_enemy(data, i, frac))  ##select enemy for class i with fraction

            pre_prune_data = np.array(pre_prune_data)


            point_class = i

            for point in same_class_data:

                #value of point except class and index
                point_val = point[1:-1]

                #check if random selcetion isn't active
                if not self.rand:
                    np.random.seed(0)

                #form same class data select some of them randomly(size of selection is frac * number of classes)
                indices = np.where(same_class_data[:, -1] == point_class)[0]
                selected_index = np.random.choice(indices, size=int(len(indices) * (frac*len(unique_class))))

                #calculate distance of same class data to selected point(remove class and index in calculation)
                same_class_distances = np.linalg.norm(same_class_data[selected_index, 1:-1] - point_val, axis=1)
                same_class_distances.sort()

                #check if duplicate point found skip that point
                if same_class_distances[0]==0:
                    K_same = same_class_distances[:K_neighbors+1]
                else:
                    K_same = same_class_distances[:K_neighbors]


                #calculate head of formula
                head = sum(K_same)

                #check if random selcetion isn't active
                if not self.rand:
                    np.random.seed(0)

                #calculate distance of opposite class data to selected point(remove class and index in calculation)
                opposite_class_distances = np.linalg.norm(pre_prune_data[:, 1:-1] - point_val, axis=1)
                opposite_class_distances.sort()

                # calculate tail of formula
                K_opp = opposite_class_distances[:K_neighbors]
                tail = sum(K_opp)

                # calculate point score
                Point_score = head / tail

                # if point score is grater than 1 it is noise point save its index in li_idx
                if Point_score > 1:
                    li_idx.append(int(point[0]))
                    continue

                #save index and Score in sc_li
                sc_li.append([int(point[0]), Point_score])

        sc_li = np.array(sc_li)

        #Remove noise point from data
        data = np.delete(data, li_idx, 0)

        #Bining score and create histogram of data
        bins = np.linspace(0, 1, 9)
        hist, bins = np.histogram(sc_li[:,1], bins=bins)

        #sort sc_li base on score
        sc_li = np.array(sorted(sc_li, key=lambda x: x[1], reverse=True))

        # calculate probability of each bins base on number of point in that bin
        p_hits = np.floor((hist / sum(hist)) * 100) / 100

        self.bins = bins
        self.hist = hist
        self.final_data = data[:, 1:]

        self.p_hits = p_hits
        self.sc_li=sc_li
        self.final_idx = li_idx

        return self

    def i_s_methods(self,method=1):
        # Choose method of reduction

        li_idx=[]
        if method == 1:
            li_idx.extend(self.__method1(self.sc_li))
        elif method == 2:
            li_idx.extend(self.__method2(self.sc_li))
        else:
            raise ("Worng value of methos")

        # Add new index to last remove index
        li_idx.extend(self.final_idx)

        return self,li_idx







