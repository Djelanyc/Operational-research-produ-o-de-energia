#!/usr/bin/env python
# coding: utf-8

# # Otimização da Produção Sustentável de energia: um Modelo de Programação Linear para Maximizar Lucros entre Energia Solar e Eólica
# Autor: Djelany Cruz\
# Data: 15 de Outubro de 2025\
# LinkedIn: www.linkedin.com/in/djelany-cruz
# 
# Ferramentas usadas: Python (PuLP, NumPy, Matplotlib, Plotly)

# ## Resumo
# Este projeto tem como objetivo otimizar a produção sustentável de energia proveniente de fontes renováveis — solar e eólica — através da aplicação de um modelo de Programação Linear. O estudo visa determinar a combinação ótima de produção diária que maximiza o lucro total da empresa, respeitando simultaneamente restrições operacionais, ambientais e de manutenção.
# 
# O modelo foi desenvolvido em Python, utilizando a biblioteca PuLP para formulação e resolução do problema, complementado por visualizações 3D e uma análise de sensibilidade para avaliar o impacto das principais restrições no lucro ótimo.
# 
# Os resultados obtidos indicam que a produção ótima é de 60 MW de energia solar e 50 MW de energia eólica, resultando num lucro máximo diário de 5.700 €, sem ultrapassar os limites de emissões e tempo de manutenção. Este estudo demonstra o potencial da Programação Linear como ferramenta de apoio à tomada de decisão sustentável e eficiente no setor energético.
# ## Abstract
# This project aims to optimize the sustainable production of energy from renewable sources — solar and wind — through the application of a Linear Programming model. The study seeks to determine the optimal daily production mix that maximizes the company’s total profit while simultaneously respecting operational, environmental, and maintenance constraints.
# 
# The model was developed in Python using the PuLP library for problem formulation and resolution, complemented by 3D visualizations and a sensitivity analysis to assess the impact of key constraints on the optimal profit.
# 
# The results indicate that the optimal production consists of 60 MW of solar energy and 50 MW of wind energy, resulting in a maximum daily profit of €5,700, without exceeding emission or maintenance time limits. This study demonstrates the potential of Linear Programming as a decision-support tool for achieving sustainable and efficient energy management.
# 

# ## Problema
#  
# 
# A empresa **de energia** pretende determinar a produção diária ótima de energia a partir de duas fontes renováveis — **energia solar**  (em MW) e **energia eólica** (em MW) — de modo a **maximizar o lucro diário total**, sujeita a restrições de capacidade, de garantia de abastecimento, de manutenção e de emissões de gases de efeito estufa. 
# Os lucros unitários são de **45 € por MW** para a energia solar e **60 € por MW** para a energia eólica. Assume-se que a produção diária em MWh corresponde a \(24x_1\) e \(24x_2\).
# 
# ---
# 
# ### Parametrização numérica adotada no modelo
# 
# - Capacidade máxima solar: **60 MW**  
# - Capacidade máxima eólica: **80 MW**  
# - Demanda mínima total: **90 MW**  
# - Tempo de manutenção disponível por dia: **70 horas**  
# - Tempo de manutenção por MW: **0,5 h/MW** para solar e **0,8 h/MW** para eólica  
# - Fator de emissão (ciclo de vida) assumido: **45 kg CO₂/MWh** para solar e **12 kg CO₂/MWh** para eólica  
# - Teto diário de emissões: \(E_{\text{max}} = 100\,000\) kg CO₂/dia  
# - Intensidade média máxima de emissões: \(I_{\text{max}} = 30\) kg CO₂/MWh
# 
# 
# 

# ## Objetivo geral
# 
# Criar um modelo de otimização linear (Programação Linear) com duas variáveis de decisão, explorado em profundidade, com:
# 
# * Modelagem matemática e solução ótima.
# 
# * Visualização do espaço viável e da função objetivo em 3D.
# 
# * Análise de sensibilidade.
# 
# * Interpretação prática dos resultados.

# ## 1- Formulação matemática do problema

# ### Variáveis de decisão
# 
#   
# x_1 = Produção de energia solar em MW
# 
# x_2 = Produção de energia eólica em MW
# 
# 
# * Função objectivo
# 
# **Maximizar lucro diário** 45 *24x_1 + 60 *24x_2
# 
# 
# * Restrições 
# 
# x_1 >= 0 - Não negatividade.\
# x_2 >= 0 - Não negatividade.\
# x_1 <= 60 MW - Potência máxima de energia solar.\
# x_2 <= 80 MW - Potência máxim de energia eólica.\
# x_1 + x_2 >= 90 MW - Procura mínima diária.\
# 0,5x_1 + 0,8x_2 <= 70 Horas - Garanria diária de manutenção.\
# 45 kg CO₂/MWh * 24x_1 MWh + 12 kg CO₂/MWh*24x_2 MWh <= 100000 kg CO₂/dia  <=> 1080x_1 kg CO₂/dia + 288x_2 kg CO₂/dia <= 100000 kg CO₂/dia - Teto diário de emissões.
# 

# ## 2-Modelagem em python

# In[17]:


# importações
import numpy as np
import matplotlib.pyplot as plt
import pulp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


# Definição de parâmetros fixos
lucro_solar = 45
lucro_eolica = 60
max_solar = 60
max_eolica = 80
demanda_min = 90
tempo_max = 70
Emax = 100000 

#  Fatores de emissão
emissao_solar = 45  #45 kg CO₂/MWh para energia solar,
emissao_eolica = 12 # 12 kg CO₂/MWh para energia eólica.
#Usados na restrição ambiental (convertidos para kg CO₂/dia multiplicando por 24 horas → 1080 e 288 no modelo).


# In[19]:


# Definição da função que resolve o problema

def resolver_modelo (max_solar, max_eolica, demanda_min, tempo_max, Emax):
    modelo = pulp.LpProblem('PL_Otimizaçao', pulp.LpMaximize)
    #Variáveis de decisão
    x_1 = pulp.LpVariable('Energia Solar', lowBound = 0)
    x_2 = pulp.LpVariable('Energia Eólica', lowBound = 0)
    #FO
    modelo += lucro_solar * x_1 + lucro_eolica * x_2
    #restrições
    modelo += x_1 <= max_solar     # Capacidade mâxima solar
    modelo += x_2 <= max_eolica    # Capacidade máxima eólica
    modelo += x_1 + x_2 >= demanda_min  # Demanda mínima de energia
    modelo += 0.5*x_1 + 0.8*x_2 <= tempo_max  # Garantia máxima diaria de manutenção
    modelo += 1080*x_1 + 288*x_2 <= Emax  # Teto de emissões
    # Resolver o problema
    modelo.solve(solver= pulp.PULP_CBC_CMD(msg= False))
    return pulp.value(modelo.objective)


# In[50]:


#Resultados
modelo = pulp.LpProblem('PL_Otimizaçao', pulp.LpMaximize)
    #Variáveis de decisão
x_1 = pulp.LpVariable('Energia Solar', lowBound = 0)
x_2 = pulp.LpVariable('Energia Eólica', lowBound = 0)
#FO
modelo += lucro_solar * x_1 + lucro_eolica * x_2
#restrições
modelo += x_1 <= max_solar     # Capacidade mâxima solar
modelo += x_2 <= max_eolica    # Capacidade máxima eólica
modelo += x_1 + x_2 >= demanda_min  # Demanda mínima de energia
modelo += 0.5*x_1 + 0.8*x_2 <= tempo_max  # Garantia máxima diaria de manutenção
modelo += 1080*x_1 + 288*x_2 <= Emax  # Teto de emissões

# Resolver o problema
modelo.solve(solver= pulp.PULP_CBC_CMD(msg= False))
# resultados
print(f'Estado do modelo: {pulp.LpStatus[modelo.status]}\n')
print(x_1.varValue, 'MWh de energia solar a produzir')
print(x_2.varValue , 'MWh de energia eólica a produzir')
print(f'Lucro máximo= {resolver_modelo(max_solar, max_eolica, demanda_min, tempo_max, Emax)}€')


# In[49]:


# verificação de cumprimento de restrições

print('Se x_1 =',x_1.varValue, 'e', 'x_2 =',x_2.varValue,
      'MW, logo a não negatividade foi respeitada.')
print('Potência máxima Solar produzida =', x_1.varValue, 
      'MW, logo a Potência máxima de 60 MW foi respeitada.')
print('Potência máxima eólica produzida =', x_2.varValue, 
      'MW, logo a Potência máxima de 80 MW foi respeitada.')
print('Produção total diaria =', x_1.varValue + x_2.varValue, 
      'MW, logo a procura mínima foi satisfeita.')
print('Manutenção diaria necessária=',0.5*x_1.varValue + 0.8*x_2.varValue,
      'horas, logo a a garantia de 70 horas de manutenção diaria não foi excedida.')
print('Total de émissões =',1080*x_1.varValue + 288*x_2.varValue, 
      'kg CO₂/dia, que não ultrapassa o teto de  100.000 kg CO₂/dia.')


# ## 3- Vizualização 3D do espaço viável 

# In[22]:


# aranges
x_vals = np.arange(0,100, 0.01) #x_1
y_vals = np.arange(0,100, 0.01) # x_2
X,Y = np.meshgrid(x_vals, y_vals)

Z = np.where((1080*X + 288*Y  <= 100000) &
             (X <= 60) &
             (Y <= 80) &
             (X + Y >= 90) &
             (0.5*X + 0.8*Y <= 70),
            45*X + 60*Y, np.nan)


# In[23]:


# ponto ótimo
max_index = np.nanargmax(Z)
x_opt, y_opt, z_opt = X.flatten()[max_index], Y.flatten()[max_index], Z.flatten()[max_index]


# In[52]:


#criar a figura para o gráfico

fig= plt.figure('Produção ótima de energias renováveis', figsize= (9,16))
plot = fig.add_subplot(111, projection='3d')
plot.text(x_opt, y_opt, z_opt, f'Lucro máximo:{z_opt}€\n Solar: {x_opt}MW \n Eólica: {y_opt}MW')

# Desenhas o gráfico
plot.plot_surface(X,Y,Z, cmap= 'hot', alpha= 0.7)

# bolinha sobre o ponto ótimo
plot.scatter(x_opt, y_opt, z_opt, color= 'red', s= 30)

# Eixos
plot.set_xlabel ('Solar (MW)')
plot.set_ylabel ('Eólica (MW)')
plot.set_zlabel ('Lucro (€)')
plot.set_title ('Produção ótima de energia')

#O gráfico
plt.show() 


# Este gráfico 3D permite visualizar  a área viável de combinações de produção (solar, eólica), ou seja, todas as combinações que caim fora do desenho do gráfico infrigem a pelo menos uma restrição , o que torna inviáveis tais produções.

# ## 4- Análise de sensibilidade

# In[47]:


# Lucro ótimo vs Teto de emissões
Emax_values = range(70000, 140001, 500)
lucro_Emax = [resolver_modelo(max_solar, max_eolica, demanda_min, tempo_max, E)
              for E in Emax_values]

# Lucro ótimo vs Tempo de manutenção garantido
tempo_m_values = range(50,141,5)
lucro_tempo = [resolver_modelo(max_solar, max_eolica, demanda_min, T, Emax)
               for T in tempo_m_values]
# Lucro ótimo vs Demanda mínima
demanda_values = range(0,201,20)
lucro_demanda =[resolver_modelo(max_solar, max_eolica, D, tempo_max, Emax)
             for D in demanda_values]
# Lucro ótimo vs capacidade solar 
solar_values = range(20,111,5)
lucro_s = [resolver_modelo(S, max_eolica, demanda_min, tempo_max, Emax)
              for S in solar_values]

#Lucro ótimo vs capacidade eólica
eolica_values = range(20,111,5)
lucro_e = [resolver_modelo(max_solar, E, demanda_min, tempo_max, Emax)
                for E in eolica_values]

#Gráficos

def grafico(x,y, xlabel, titulo):
    plt.figure(figsize=(7,4))
    plt.plot(x,y, marker ='o',color= 'r',linewidth=2)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel('Lucro Ótimo (€)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plt.rcParams['font.family'] = 'DejaVu Sans'    


grafico(Emax_values,lucro_Emax,'Emissões kg CO₂/dia','Lucro ótimo vs Teto de emissões')
grafico(tempo_m_values,lucro_tempo,'Tempo de manutenção (h)', 'Lucro ótimo vs Tempo de manutenção garantido')
grafico(demanda_values, lucro_demanda,'Demanda mínima (MW)','Lucro ótimo vs Demanda mínima')
grafico(solar_values, lucro_s, 'Capacidade  de produção solar (MW)','Lucro ótimo vs capacidade solar')
grafico(eolica_values,lucro_e,'Capacidade  de produção eólica (MW)','Lucro ótimo vs capacidade eólica')


#   Os gráficos de linha acima, mostram-nos o impacto linear, no lucro ótimo que a variação de cada restrição do modelo apresenta. 
# 
# 
# >O teto de emissões impacta razoavelmente o lucro ótimo da produção, ddeixando de ser impactante a partir dos 80000 kgCO2/dia. Isto significa que não é possível aumentar o lucro ótimo da produção elevando o teto de emissões de carbono pois o lucro ótimo implica emissões de 79200.0 kg CO₂/dia.
# 
# 
# >Já o tempo de manutenção parece muito mais relevante. O intervalo mais importante nesta relação entre o tempo de manutenção e o lucro Ótimo, situa-se entre as 55h as 95h, sendo ela linear com um declive positivo, isto é, qunato maior for o número de horas de manutenção diárias garantida, maior será o lucro, podendo atingir o máximo de 7500€ (+1800€ que o ótimo atual).
# 
# >A variação demanda mínima não influencia de forma alguma o lucro ótimo da empresa.
# 
# >O lucro ótimo da empresa, mostra-se de certa forma sensível a capacidade de produção solar até aos 85MW de capacidade, gerando 5873.3€ (+173.3€ que o ótimo atual), acima disto deixa de gerar mais lucro.
# 
# >A capacidade de produção de energia elólica mostra-se bastante relevante na ótica da conservação dos ganhos de lucro ótimo pois cada 1MW de capacidade perdido pode representar uma perda significativa, porém, por outro lado não apresenta o mesmo comportamento, quer dizer, o aumento da capacidade atual não impacta os ganhos.

# 
# ## 5- Análise dos resultados

# Perante aos desafios apresentados pelas restrições neste problema, o modelo foi capaz de encontrar uma solução ótima diaria para a produção de energia (**60MW de energia Solar** e **50 MW de energia Eólica**) que geram um lúcro máximo de **5700€**/dia.
# Foi possível determinar também quais restrições merecem mais atenção na ótica do aumento dos lucros, ou seja as que mais travam os ganhos, elas são: o **tempo de manutenção diari**o com uma margem de **1800€** acima ótimo atual, e a **capacidade de produção de energia solar** com uma margem de **173.3€** acima ótimo atual.
# 
# É importante salientar que para além dos obstáculos sobre a produção de energia desta empresa, muitos outros obstáculos podem surgir e todo eles deverm ser analisados para um atomada de decisão msis informada, cuidada, real chegando a soluções cada veis mais reais no dia a dia da produção.
# 
# O modelo proposto demonstra como técnicas de Programação Linear podem apoiar decisões sustentáveis e economicamente eficientes no setor energético.
