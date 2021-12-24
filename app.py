import streamlit as st
import time
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from helper_functions import *
from matplotlib import gridspec

st.set_page_config(
    page_title='Q-Learning App',
    page_icon='🤖', layout="centered")
st.title('Q-Learning Model Demo')

st.sidebar.subheader('Set-up:')
speed = st.sidebar.selectbox(
    label='Speed of the simulation:',
    options=('Slow', 'Medium', 'Fast'),
    index=1)
n_total = st.sidebar.slider('Number of trials (T):', min_value=50, max_value=500, step=50, value=200)

SEED = st.sidebar.number_input(
    label='Seed value:', min_value=1,
    max_value=1000, value=42, step=1,
    help='Each value of random seed produces a unique set of probability values.')
rnd_generator = np.random.default_rng(SEED)

if st.sidebar.checkbox(
        'Apply seed also for the agent choices?',
        help='If selected, the results will be the same for each simulation.'):
    rnd_generator_choice = rnd_generator
else:
    rnd_generator_choice = None

option = st.sidebar.selectbox(
    'Outcome probability policy:',
    ('Stochastic', 'Constant', 'Constant with switch at ⌈T/2⌉'))

if option in ['Constant', 'Constant with switch at ⌈T/2⌉']:
    if st.sidebar.checkbox(
            'Specify probabilities manually?',
            help='Please, select also "Freeze probability values?" when selecting this.'):
        helper_text = """
        Input probability values in a Python list format
        (for example, [0.25, 0.57, 0.82]).
        """
        probs = st.sidebar.text_input(
                label='Probabilities of reward:',
                help=helper_text,
                value=str(list(rnd_generator.random(size=(3,)).round(2))))

        # create a matrix of shape (n_trials, n_arms)
        probs = ast.literal_eval(probs) # convert str to floats
        probs = np.repeat(
            np.reshape(probs, (1, 3)),
            n_total, axis=0)

        if option == 'Constant with switch at ⌈T/2⌉':
            n_switch = int(np.floor(n_total/2))
            probs_switch = st.sidebar.text_input(
                    label=f'Probabilities of reward (after {n_switch} trials):',
                    help=helper_text,
                    value=str(list(rnd_generator.random(size=(3,)).round(2))))

            probs_switch = ast.literal_eval(probs_switch)
            probs[n_switch:] = np.repeat(
                np.reshape(probs_switch, (1, 3)),
                n_total-n_switch, axis=0)
    else:
        probs = np.repeat(
            rnd_generator.random(size=(1, 3)),
            n_total, axis=0)
        if option == 'Constant with switch at ⌈T/2⌉':
            n_switch = int(np.floor(n_total/2))
            probs_switch = np.repeat(
                rnd_generator.random(size=(1, 3)), n_total-n_switch, axis=0)
            probs[n_switch:] = probs_switch

else:

    smoothness = st.sidebar.slider(
        label='"Smoothness" of the probability change:',
        min_value=1, max_value=10, step=1, value=5,
        help='Lower values result in a higher deviation, whereas higher values result in a more smooth curve.')

    scaler = MinMaxScaler(feature_range=(0, 1))
    probs = rnd_generator.normal(size=(n_total, 3)).cumsum(axis=0)
    probs = gaussian_filter1d(probs, sigma=smoothness, axis=0)
    probs = scaler.fit_transform(probs) # set to be in a [0,1] range

reward_val = st.sidebar.slider(
    'Reward value:', min_value=0,
    max_value=10, step=1, value=1)

punish_val = st.sidebar.slider(
    'Punishment value:', min_value=-10,
    max_value=0, step=1, value=-1)

plot_q = st.sidebar.checkbox('Include plot for value functions?', value=True)

n_agents = st.sidebar.slider(
    'Number of agents to run:', min_value=1,
    max_value=4, step=1, value=2)

params_dict = {}

for i_agent in range(n_agents):
    st.sidebar.subheader(f'Agent #{i_agent+1} parameters:')

    params_dict[i_agent] = {
        'alpha_pos': st.sidebar.slider(
            label='Learning rate positive (α_pos):', key=f'alpha_pos_{i_agent+1}',
            min_value=0.00, max_value=1.0, step=0.05, value=0.25),
        'alpha_neg': st.sidebar.slider(
            label='Learning rate negative (α_neg):', key=f'alpha_neg_{i_agent+1}',
            min_value=0.00, max_value=1.0, step=0.05, value=0.25),
        'beta': st.sidebar.slider(
            label='Inverse temperature (β):', key=f'beta_{i_agent+1}',
            min_value=0.25, max_value=10.0, step=0.25, value=1.0),
        'R': st.sidebar.slider(
            label='Reward sensitivity (R):', key=f'R_{i_agent+1}',
            min_value=0.0, max_value=5.0, step=0.5, value=1.0),
        'P': st.sidebar.slider(
            label='Punishment sensitivity (P):', key=f'P_{i_agent+1}',
            min_value=0.0, max_value=5.0, step=0.5, value=1.0),
        'decay_rate': st.sidebar.slider(
            label='Decay rate (α):', key=f'alpha_{i_agent+1}',
            min_value=0.00, max_value=1.0, step=0.05, value=0.00,
            help='Value 0 results in no forgetting, whereas value 1 results in setting value function of unselected '+
                 'options to zero.')
    }

with open('description.md', mode='r') as f:
    desc_text = f.read()

st.markdown(desc_text)
colors_opt = ['#82B223', '#2EA8D5', '#F5AF3D']

E_r = (probs * reward_val) + (1 - probs) * punish_val # expected value of an outcome
best_option = E_r.argmax(axis=1) + 1

fig = plt.figure(figsize=(10, 3))
ax1 = fig.subplots()
for i in range(2, -1, -1):
    ax1.plot(probs[:, i], c=colors_opt[i], label=f'#{i+1}', lw=4)
ax1.set_xlabel('Trial', fontweight='bold')
ax1.set_ylabel('P(Reward)', fontweight='bold')
ax1.set_title(
    'Probability of the Reward\nand Expected Value of the Outcome',
    fontweight='bold', fontsize=15,)
ax1.legend(title="Option")

# second y-axis on the right side
ax2 = ax1.twinx()
ax2.plot(E_r)
for i in range(2, -1, -1):
    ax2.plot(E_r[:, i], c=colors_opt[i])
ax2.set_xlabel('Trial', fontweight='bold')
ax2.set_ylabel('E[Outcome]', fontweight='bold')

st.pyplot(fig)

if st.button('Run the simulation!'):

    # specify the speed of simulations
    if speed == 'Slow':
        s = 100 # number of steps to finish simulation for 1 agent
    elif speed == 'Medium':
        s = 30
    else:
        s = 3

    batch_size = int(np.ceil(n_total/s))
    sample_indexes = np.arange(0, n_total+batch_size, step=batch_size)

    # status bar init
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i_agent in range(n_agents):
        actions, gain, Qs = generate_agent_data(
            alpha_pun=params_dict[i_agent]['alpha_neg'],
            alpha_rew=params_dict[i_agent]['alpha_pos'],
            rew_sens=params_dict[i_agent]['R'],
            pun_sens=params_dict[i_agent]['P'],
            decay_rate=params_dict[i_agent]['decay_rate'],
            beta=params_dict[i_agent]['beta'],
            rew_prob=probs,
            reward_val=reward_val,
            punish_val=punish_val,
            n_trials=n_total,
            n_arms=3,
            rnd_generator=rnd_generator_choice
        )

        agent_data = pd.DataFrame({
            'subjID': f'Agent #{i_agent+1}', 'trial': range(1, n_total+1),
            'choice': actions+1, 'gain': gain})

        fig = plt.figure(figsize=(10, 4))
        if plot_q:
            gs = gridspec.GridSpec(3, 2, width_ratios=[5, 2], height_ratios=[2, 5, 3])
        else:
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 2], height_ratios=[2, 5])
        gs.update(wspace=0.1, hspace=0.7)
        fig.suptitle(f'Agent #{i_agent+1} Performance', fontweight='bold')
        ax1 = plt.subplot(gs[0])
        ax1.set_ylim(-50, 50)
        ax1.set_xlim(0, n_total+1)
        plot1, = ax1.plot(
            agent_data['trial'][:2],
            agent_data['gain'][:2].cumsum(), color='#4000AF')
        ax1.axhline(y=0, lw=0.5, linestyle='--', color='#FF7777')
        ax1.spines[:].set_visible(False)
        ax1.spines['left'].set_visible(True)
        ax1.set_xticks([])
        ax1.set_ylabel('Cumulative\nReward', fontweight='bold')

        ax2 = plt.subplot(gs[2])
        ax2.set_ylim(0.5, 3.5)
        ax2.set_xlim(0, n_total+1)
        # ax2.plot(best_option[:i], 'o', c='red', alpha=0.2)
        plot2, = ax2.plot(
            agent_data['trial'][:2],
            agent_data['choice'].astype(str)[:2],
            'o--', color='black', lw=0.5)
        if option == 'Constant with switch at ⌈T/2⌉':
            ax2.axvline(
                n_switch, lw=0.5, linestyle='--',
                color='red', label='Switch')
            ax2.legend()
        ax2.set_ylabel('Choice', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        if plot_q:
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xticks([])
        else:
            ax2.set_xlabel('Trial', fontweight='bold')
        ax2.set_yticks([1, 2, 3], ["#1", "#2", "#3"])

        ax3 = plt.subplot(gs[3])
        ax3.set_ylim(0.5, 3.5)
        ax3.set_xlim(0, agent_data['choice'].value_counts().max()+5)
        z = agent_data[:2].groupby(['choice'], as_index=False)['subjID'].count()
        z.rename(columns={'subjID': 'count'}, inplace=True)
        for i_option in range(1, 4):
            if z['choice'].isin([i_option]).sum() == 0:
                z = z.append(
                    pd.DataFrame({'choice': [i_option], 'count': 0}),
                    ignore_index=True)\
                    .sort_values(by='choice')
        plot3 = ax3.barh(z['choice'], z['count'], color=colors_opt)
        ax3.patch.set_visible(False)
        ax3.axis('off')

        if plot_q:
            ax4 = plt.subplot(gs[4])
            ax4.set_xlim(0, n_total+1)
            ax4.set_ylim(
                punish_val*params_dict[i_agent]['P']*1.1,
                reward_val*params_dict[i_agent]['R']*1.1)
            for i_option in range(3):
                ax4.plot(
                    agent_data['trial'][:2],
                    Qs[:2, i_option],
                    color=colors_opt[i_option],
                    label=f'#{i_option+1}')
            if option == 'Constant with switch at ⌈T/2⌉':
                ax4.axvline(
                    n_switch, lw=0.5, linestyle='--',
                    color='red', label='Switch')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.set_xlabel('Trial', fontweight='bold')
            ax4.set_ylabel('Value\nFunction', fontweight='bold')

        the_plot = st.pyplot(plt)

        def animate(i):
            ax1.set_ylim(
                agent_data['gain'][:i].cumsum().min()-20,
                agent_data['gain'][:i].cumsum().max()+20)
            plot1.set_data(agent_data['trial'][:i], agent_data['gain'][:i].cumsum())
            # ax2.plot(best_option[:i], 'o', c='red', alpha=0.2)
            plot2.set_data(agent_data['trial'][:i], agent_data['choice'][:i])
            z = agent_data[:i].groupby(['choice'], as_index=False)['subjID'].count()
            z.rename(columns={'subjID': 'count'}, inplace=True)
            for i_option in range(1, 4):
                if z['choice'].isin([i_option]).sum() == 0:
                    z = z.append(
                        pd.DataFrame({'choice': [i_option], 'count': 0}),
                        ignore_index=True)\
                        .sort_values(by='choice')
            ax3.barh(z['choice'], z['count'], color=colors_opt)
            if plot_q:
                for i_opt in range(3):
                    ax4.plot(
                        agent_data['trial'][:i], Qs[:i, i_opt],
                        color=colors_opt[i_opt])
            the_plot.pyplot(plt)

        for i_batch in range(1, len(sample_indexes)):
            animate(sample_indexes[i_batch])
            indx = int(i_agent*100/n_agents) + int(i_batch*100/s / n_agents)
            status_text.text("%i%% Complete" % indx)
            progress_bar.progress(indx)
            time.sleep(0.001)

    status_text.text("%i%% Complete" % 100)
    progress_bar.progress(100)
