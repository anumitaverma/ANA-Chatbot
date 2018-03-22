import csv

class chatbot():

    def __init__(self):
        self.training_set = []
        self.normalized_data=[]

    def input(self):

        with open('training.csv','r',encoding="utf8") as f:
            lines=csv.reader(f)
            count=0
            for row in lines:
                count=count+1
        f.close()


        with open('training.csv','r',encoding="utf8") as f:
            lines=csv.reader(f)
            prev=""
            c=0
            for row in lines:
                if(row==prev):
                    prev=row
                    continue
                self.training_set.append(row)
                c=c+1
                prev=row
            print(self.training_set[1])

        self.normalized_data=[[] for j in range(c)]

        f.close()
        #lowercase
        i=0
        for row in self.training_set:
            for elements in row[0:3]:
                self.normalized_data[i].append(elements.lower())
            i=i+1

        #null
        for row in self.normalized_data:
            for i,elements in enumerate(row[0:3]):
                if(elements==''):
                    row[i]='null'

        for row in self.normalized_data:
            print(row)

        print(len(self.normalized_data))
        print(str(count) + " " +str(c))

if __name__ == '__main__':
    cb =chatbot()
    cb.input()