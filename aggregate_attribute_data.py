import pandas as pd

#path of file
path='C:\Users\Arslan Qadri\Google Drive\Sem 2\Stats Learning\project\\'


#train=pd.read_csv(path+'train.csv')
att=pd.read_csv(path+'attributes.csv',dtype='unicode',low_memory=True)   #attributes file
#desc=pd.read_csv(path+'product_descriptions.csv')

id=att['product_uid']

# convert uid to string
id_unique=set([int(str(i)) for i in id if len(str(i))>4])


o=0
item=[]
for i in id_unique:
    v=[]
    df=att[att['product_uid']==str(i)]
    df.reset_index(inplace=True)
    for k,j in enumerate(df):
        
        atts= df['name'][k] +' : '+str(df['value'][k])
        

        v.append(atts)
	# for testing status	
    o+=1
    if o % 100 ==0:
        print 'Processed ',o,' of',len(id_unique),' items'  
    item.append((i,v))

it=pd.DataFrame(item,columns=['product_uid','attributes'])

#print to file
it.to_csv('edit.csv')

