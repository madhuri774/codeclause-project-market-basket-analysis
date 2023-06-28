#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().system('pip install apyori')


# In[35]:


import pandas as pd
import numpy as np
from apyori import apriori


# In[36]:


df = pd.read_csv('Market_Basket_Optimisation.csv',header=None)


# In[5]:


df.head()


# In[6]:


df.fillna(0,inplace=True)


# In[7]:


df.head()


# In[8]:


transactions = []
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])


# In[9]:


transactions[0]


# In[10]:


rules = apriori(transactions,min_support=0.003,min_confidance=0.2,min_lift=3,min_length=2)


# In[12]:


rules


# In[13]:


Results = list(rules)
Results


# In[14]:


df_results = pd.DataFrame(Results)


# In[15]:


df_results.head()


# In[21]:


#keep support in a separate data frame so we can use later.. 
support = df_results.support


# In[22]:


first_values = []
second_values = []
third_values = []
fourth_value = []

for i in range(df_results.shape[0]):
    single_list = df_results['ordered_statistics'][i][0]
    first_values.append(list(single_list[0]))
    second_values.append(list(single_list[1]))
    third_values.append(single_list[2])
    fourth_value.append(single_list[3])


# In[23]:


lhs = pd.DataFrame(first_values)
rhs= pd.DataFrame(second_values)
confidance=pd.DataFrame(third_values,columns=['Confidance'])
lift=pd.DataFrame(fourth_value,columns=['lift'])


# In[19]:


df_final = pd.concat([lhs,rhs,support,confidance,lift], axis=1)
df_final


# In[24]:


'''
 we have some of place only 1 item in lhs and some place 3 or more so we need to a proper represenation for User to understand. 
 removing none with ' ' extra so when we combine three column in 1 then only 1 item will be there with spaces which is proper rather than none.
 example : coffee,none,none which converted to coffee, ,
'''
df_final.fillna(value=' ', inplace=True)


# In[26]:


df_final.columns = ['lhs',1,2,'rhs','support','confidance','lift',3]


# In[27]:


df_final['lhs'] = df_final['lhs']+str(", ")+df_final[1]+str(", ")+df_final[2]


# In[28]:


df_final.drop(columns=[1,2],inplace=True)


# In[29]:


#this is final output.. you can sort based on the support lift and confidance..
df_final.head()


# In[30]:


'''
load apriori and association package from mlxtend. 
Used different dataset because mlxtend need data in below format. 

             itemname  apple banana grapes
transaction  1            0    1     1
             2            1    0     1  
             3            1    0     0
             4            0    1     0
             
 we could have used above data as well but need to perform operation to bring in this format instead of that used seperate data only.            
'''


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df1 = pd.read_csv('data.csv', encoding="ISO-8859-1")
df1.head()


# In[31]:


df1.Country.value_counts().head(5)


# In[32]:


df1 = df1[df1.Country == 'France']


# In[33]:


df1['Description'] = df1['Description'].str.strip()


# In[34]:


#some of transaction quantity is negative which can not be possible remove that.
df1 = df1[df1.Quantity >0]


# In[35]:


df1[df1.Country == 'France'].head(10)


# In[36]:


#convert data in format which it require converting using pivot table and Quantity sum as values. fill 0 if any nan values

basket = pd.pivot_table(data=df1,index='InvoiceNo',columns='Description',values='Quantity', \
                        aggfunc='sum',fill_value=0)


# In[37]:


basket.head()


# In[38]:


#this to check correctness after binning it to 1 at below code..
basket['ALARM CLOCK BAKELIKE RED'].head(10)


# In[39]:


# we dont need quantity sum we need either has taken or not so if user has taken that item mark as 1 else he has not taken 0.

def convert_into_binary(x):
    if x > 0:
        return 1
    else:
        return 0


# In[40]:


basket_sets = basket.applymap(convert_into_binary)


# In[41]:


# above steps we can same item has quantity now converted to 1 or 0.
basket_sets['ALARM CLOCK BAKELIKE RED'].head()


# In[42]:


#remove postage item as it is just a seal which almost all transaction contain. 
basket_sets.drop(columns=['POSTAGE'],inplace=True)


# In[ ]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)


# In[44]:


#it will generate frequent itemsets using two step approch
frequent_itemsets


# In[45]:


# we have association rules which need to put on frequent itemset. here we are setting based on lift and has minimum lift as 1
rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_mlxtend.head()


# In[46]:


# rules_mlxtend.rename(columns={'antecedents':'lhs','consequents':'rhs'})

# as based business use case we can sort based on confidance and lift.
rules_mlxtend[ (rules_mlxtend['lift'] >= 4) & (rules_mlxtend['confidence'] >= 0.8) ]


# In[22]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_graph(rules, rules_to_show,rules_mlxtend=None):
    G1 = nx.DiGraph()
    color_map = []
    N = 50
    colors = np.random.rand(N)
    strs = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']

    for i in range(rules_to_show):
        G1.add_node("R"+str(i))
        for a in rules.iloc[i]['antecedents']:
            G1.add_node(a)
            G1.add_edge(a, "R"+str(i), color=colors[i], weight=2)
        for c in rules.iloc[i]['consequents']:
            G1.add_node(c)
            G1.add_edge("R"+str(i), c, color=colors[i], weight=2)

    for node in G1:
        found_a_string = False
        for item in strs:
            if node == item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u, v in edges]
    weights = [G1[u][v]['weight'] for u, v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, node_color=color_map, edge_color=colors, width=weights, font_size=16,
            with_labels=False)

    for p in pos:
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    plt.show()
    draw_graph(rules_mlxtend, 10)

