import csv
import plotly 
import pandas as pd 
import plotly.graph_objs as go

# filename="loss.csv"

# with open(filename, 'r') as csvfile: 
#     # creating a csv reader object 
#     csvreader = csv.reader(csvfile)
#     for row in csvreader: 
#         rows.append(row)
    
loss=pd.read_csv("loss.csv")

trace1 = go.Scatter(
                    x = loss.epoch,
                    y = loss.loss_l1,
                    mode = "lines",
                    name = "Loss L1",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= 'lossl1')
# Creating trace2
trace2 = go.Scatter(
                    x = loss.epoch,
                    y = loss.loss_l2,
                    mode = "lines",
                    name = "Loss L2",
                    marker = dict(color = 'rgba(255, 63, 20, 0.7)'),
                    text= 'lossl2')
trace3 = go.Scatter(
                    x = loss.epoch,
                    y = loss.gan_loss,
                    mode = "lines",
                    name = "Gan Loss",
                    marker = dict(color = 'rgba(20, 52, 255, 1)'),
                    text= 'gan loss')
data = [trace1, trace2,trace3]
layout = dict(title = 'Pix2Pix Loss Graph',
              xaxis= dict(title= 'Epoch',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
plotly.offline.iplot(fig)

