import numpy as np
import scipy.io
import math
from glob import glob
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import CMRversions.probCMR as probCMR

class CMR2Reminder(probCMR.CMR2):
    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. 
        """
        self.beta_in_play = self.params['beta_rec']  
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # initialize list to store recalled items
        recalled_items = []
        support_values = []

        
        # MODIFIED: track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)
        self.torecall = np.zeros([1, nitems_in_session], dtype=np.float32)
        self.torecall[0][thislist_pres_indices] = 1
          
        if self.recall_mode == 0: # simulations 
            continue_recall = 1
            
            # start the recall with cue position if not negative
            if self.params['cue_position'] >= 0:
                self.c_net = self.c_cue.copy()
       
            # simulate a recall session/list 
            while continue_recall == 1:

                # get item activations to input to the accumulator
                f_in = self.obtain_f_in() # obtain support for items

                # recall process:
                winner_idx, support = self.retrieve_next(f_in.T)
                winner_ID = np.sort(self.all_session_items)[winner_idx]

                # If an item was retrieved, recover the item info corresponding
                # to the activation value index retrieved by the accumulator
                if winner_idx is not None:

                    recalled_items.append(winner_ID)
                    support_values.append(support)

                    # reinstantiate this item
                    self.present_item(winner_idx)

                    # MODIFIED: update the to-be-recalled remaining items
                    self.torecall[0][winner_idx] = -1

                    # update context
                    self.update_context_recall()
       
                else:
                    continue_recall = 0

                

        else: # calculate probability of a known recall list
            thislist_recs = self.recs_list_nos[self.list_idx]
            thislist_recs_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_recs)
            recall_length = np.count_nonzero(thislist_recs)
            for i in range(recall_length):
                recall_idx = thislist_recs_indices[i]
                if recall_idx < len(self.all_session_items_sorted):
                    f_in = self.obtain_f_in() 
                    
                    self.lkh += self.retrieve_probability(f_in.T,recall_idx,0) # recall process
                    # reinstantiate this item
                    self.present_item(recall_idx)
                    # update the to-be-recalled remaining items
                    self.torecall[0][recall_idx] = -1

                    
                    # update context
                    self.update_context_recall()

                else:
                    self.count += 1

            f_in = self.obtain_f_in() 
            self.lkh += self.retrieve_probability(f_in.T,0,1) # stopping process

        # update counter of what list we're on: commmented out because it is done at the end of reminder session instead
        # self.list_idx += 1

        return recalled_items, support_values

    
    def reminder_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. 
        """
        self.beta_in_play = self.params['beta_rec']  
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # MODIFIED: track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)
        thislist = thislist_pattern.tolist()
        
        recalled = self.torecall.copy()
        reminders = [i for i in range(nitems_in_session) if recalled[0][i]==1 ]
        recalls_sp = []
        reminders_sp = []
        accs = []
        for i in range(len(reminders)): 
            acc = []
            reminder = reminders[i]
            reminder_ID = np.sort(self.all_session_items)[reminder]
            reminders_sp.append(thislist.index(reminder_ID))

            for r in range(20): # repetitions for each reminder
                self.torecall = recalled.copy() 
                self.present_item(reminder)
                self.torecall[0][reminder] = -1 # not counting the reminder itself
                self.beta_in_play = 1 # fully reset the context given the reminder
                self.update_context_recall()
                self.beta_in_play = self.params['beta_rec'] # set back to regular beta during recall

                # initialize list to store recalled items
                recalled_items = []

                if self.recall_mode == 0: # simulations 
                    continue_recall = 1
                    # simulate a recall session/list 
                    while continue_recall == 1:

                        # get item activations to input to the accumulator
                        f_in = self.obtain_f_in() # obtain support for items

                        # recall process:
                        winner_idx, support = self.retrieve_next(f_in.T)
                        winner_ID = np.sort(self.all_session_items)[winner_idx]

                        # If an item was retrieved, recover the item info corresponding
                        # to the activation value index retrieved by the accumulator
                        if winner_idx is not None:

                            #recalled_items.append(winner_ID)
                            recalled_items.append(thislist.index(winner_ID))

                            # reinstantiate this item
                            self.present_item(winner_idx)

                            # MODIFIED: update the to-be-recalled remaining items
                            self.torecall[0][winner_idx] = -1

                            # update context
                            self.update_context_recall()

                        else:
                            continue_recall = 0
                acc.append(len(recalled_items))
            accs.append(acc)    
            recalls_sp.append(recalled_items) # only store the last repetition

        context_tem = self.M_CF_tem[:,thislist_pres_indices]
        context_sem = self.M_CF_sem[:,thislist_pres_indices]
        #contexts_IN = np.append(contexts_IN, self.c_in_normed,axis=1)
        #contexts_drift = np.append(contexts_drift, self.c_net,axis=1)         
        pca = PCA(n_components=2)
        contexts_tem = pca.fit_transform(context_tem.T)
        contexts_sem = pca.fit_transform(context_sem.T)
        contexts_pca = np.concatenate((contexts_tem, contexts_sem), axis=1)
        # update counter of what list we're on
        self.list_idx += 1

        return reminders_sp,recalls_sp,accs,contexts_pca

