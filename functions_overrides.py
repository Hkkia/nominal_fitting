import numpy as np
import time

from probCMR_overrides import CMR2Reminder
from CMRversions.functions import init_functions
Functions = init_functions(CMR2Reminder)

class FunctionsReminder(Functions):
    def run_CMR2_singleSubj(self, recall_mode, pres_sheet, rec_sheet, LSA_mat, params):

        """Run CMR2 for an individual subject / data sheet"""

        # init. lists to store CMR2 output
        resp_values = []
        support_values = []
        reminders_values = []
        recalls_values = []
        accs_values = []
        pcas_values = []

        # create CMR2 object
        this_CMR = CMR2Reminder(
            recall_mode=recall_mode, params=params,
            LSA_mat=LSA_mat, pres_sheet = pres_sheet, rec_sheet  =rec_sheet)

        # layer LSA cos theta values onto the weight matrices
        this_CMR.create_semantic_structure()

        # Run CMR2 for each list
        for i in range(len(this_CMR.pres_list_nos)):
            # present new list
            this_CMR.present_list()

            # recall session
            rec_items_i, support_i = this_CMR.recall_session()

            # reminder session
            reminders, recalls, accs,pcas = this_CMR.reminder_session()

            # append recall responses & times
            resp_values.append(rec_items_i)
            support_values.append(support_i)
            reminders_values.append(reminders)
            recalls_values.append(recalls)
            accs_values.append(accs)
            pcas_values.append(pcas)

        return resp_values, support_values, this_CMR.lkh, reminders_values, recalls_values, accs_values, pcas_values


    def run_CMR2(self, recall_mode, LSA_mat, data_path, rec_path, params, sep_files,
                 filename_stem="", subj_id_path="."):
        """Run CMR2 for all subjects

        time_values = time for each item since beginning of recall session

        For later zero-padding the output, we will get list length from the
        width of presented-items matrix. This assumes equal list lengths
        across Ss and sessions, unless you are inputting each session
        individually as its own matrix, in which case, list length will
        update accordingly.

        If all Subjects' data are combined into one big file, as in some files
        from prior CMR2 papers, then divide data into individual sheets per subj.

        If you want to simulate CMR2 for individual sessions, then you can
        feed in individual session sheets at a time, rather than full subject
        presented-item sheets.
        """

        now_test = time.time()

        # set diagonals of LSA matrix to 0.0
        np.fill_diagonal(LSA_mat, 0)

        # init. lists to store CMR2 output
        resp_vals_allSs = []
        support_vals_allSs = []
        reminders_values_allSs = []
        recalls_values_allSs = []
        accs_values_allSs = []
        pcas_values_allSs = []
        lkh = 0

        # Simulate each subject's responses.
        if not sep_files:

            # divide up the data
            subj_presented_data, subj_recalled_data, unique_subj_ids = self.separate_files(
                data_path, rec_path, subj_id_path)

            # get list length
            listlength = subj_presented_data[0].shape[1]

            # for each subject's data matrix,
            for m, pres_sheet in enumerate(subj_presented_data):
                rec_sheet = subj_recalled_data[m]
                subj_id = unique_subj_ids[m]
                # print('Subject ID is: ' + str(subj_id))

                resp_Subj, support_Subj, lkh_Subj, reminders_values, recalls_values, accs_values, pcas_values = self.run_CMR2_singleSubj(
                            recall_mode, pres_sheet, rec_sheet, LSA_mat,
                            params)

                resp_vals_allSs.append(resp_Subj)
                support_vals_allSs.append(support_Subj)
                reminders_values_allSs.extend(reminders_values)
                recalls_values_allSs.extend(recalls_values)
                accs_values_allSs.extend(accs_values)
                pcas_values_allSs.extend(pcas_values)
                lkh += lkh_Subj
        # If files are separate, then read in each file individually
        else:

            # get all the individual data file paths
            indiv_file_paths = glob(data_path + filename_stem + "*.mat")

            # read in the data for each path & stick it in a list of data matrices
            for file_path in indiv_file_paths:

                data_file = scipy.io.loadmat(
                    file_path, squeeze_me=True, struct_as_record=False)  # get data
                data_mat = data_file['data'].pres_itemnos  # get presented items

                resp_Subj, support_Subj, lkh_Subj, reminders_values, recalls_values, accs_values, pcas_values = self.run_CMR2_singleSubj(
                    recall_mode, data_mat, LSA_mat,
                    params)

                resp_vals_allSs.append(resp_Subj)
                support_vals_allSs.append(support_Subj)
                reminders_values_allSs.extend(reminders_values)
                recalls_values_allSs.extend(recalls_values)
                accs_values_allSs.extend(accs_values)
                pcas_values_allSs.extend(pcas_values)
                lkh += lkh_Subj

            # for later zero-padding the output, get list length from one file.
            data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                         struct_as_record=False)
            data_mat = data_file['data'].pres_itemnos

            listlength = data_mat.shape[1]


        ##############
        #
        #   Zero-pad the output
        #
        ##############

        # If more than one subject, reshape the output into a single,
        # consolidated sheet across all Ss
        if len(resp_vals_allSs) > 0:
            resp_values = [item for submat in resp_vals_allSs for item in submat]
            support_values = [item for submat in support_vals_allSs for item in submat]
        else:
            resp_values = resp_vals_allSs
            support_values = support_vals_allSs

        # set max width for zero-padded response matrix
        maxlen = listlength * 1

        nlists = len(resp_values)

        # init. zero matrices of desired shape
        resp_mat    = np.zeros((nlists, maxlen))
        support_mat = np.zeros((nlists, maxlen))


        # place output in from the left
        for row_idx, row in enumerate(resp_values):

            resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
            support_mat[row_idx][:len(row)]   = support_values[row_idx]


        #print('Analyses complete.')

        #print("CMR Time: " + str(time.time() - now_test))
        return resp_mat, support_mat, lkh, reminders_values_allSs, recalls_values_allSs, accs_values_allSs, pcas_values_allSs
    
    def model_probCMR(self, N, ll, lag_examine,data_id):  
        """Error function that we want to minimize"""
        ###############
        #
        #   simulate free recall data
        #
        # N: 0 - obtain lists of recall, in serial positions
        # N: 1 - obtain likelihood given data
        # N >1 - plot behavioral data with error bar with N being the number of times in simulations
        #
        # ll: list length (ll=16)
        #
        # lag_examine: lag used in plotting CRP
        #
        ###############
        LSA_mat, data_path, data_rec_path, data_cat_path, subjects_path = self.load_data(data_id)
            
        data_pres = np.loadtxt(data_path, delimiter=',')
        data_rec = np.loadtxt(data_rec_path, delimiter=',')


        # model parameters
        if data_id==0:
            param_dict = {

                'beta_enc':  0.3187893806764954,           # rate of context drift during encoding
                'beta_rec':  0.9371120781560975,           # rate of context drift during recall
                'beta_rec_post': 1,      # rate of context drift between lists
                                                # (i.e., post-recall)

                'gamma_fc': 0.1762454837715133,  # learning rate, feature-to-context
                'gamma_cf': 0.5641689110824742,  # learning rate, context-to-feature
                'scale_fc': 1 - 0.1762454837715133,
                'scale_cf': 1 - 0.5641689110824742,


                's_cf': 0.8834467032413329,       # scales influence of semantic similarity
                                        # on M_CF matrix

                's_fc': 0.0,            # scales influence of semantic similarity
                                        # on M_FC matrix.
                                        # s_fc is first implemented in
                                        # Healey et al. 2016;
                                        # set to 0.0 for prior papers.

                'phi_s': 2.255110764387116,      # primacy parameter
                'phi_d': 0.4882977227079478,      # primacy parameter


                'epsilon_s': 0.0,      # baseline activiation for stopping probability 
                'epsilon_d': 2.2858636787518285,        # scale parameter for stopping probability 

                'k':  6.744153399759922,        # scale parameter in luce choice rule during recall

                # parameters specific to optimal CMR:
                'primacy': 0.0,
                'enc_rate': 1.0,

            }
        elif data_id==1:
            param_dict = {

                'beta_enc':  0.7887626184661226,           # rate of context drift during encoding
                'beta_rec':  0.49104864172027485,           # rate of context drift during recall
                'beta_rec_post': 1,      # rate of context drift between lists
                                                # (i.e., post-recall)

                'gamma_fc': 0.4024001271645564,  # learning rate, feature-to-context
                'gamma_cf': 1,  # learning rate, context-to-feature
                'scale_fc': 1 - 0.4024001271645564,
                'scale_cf': 0,


                's_cf': 0.8834467032413329,       # scales influence of semantic similarity
                                        # on M_CF matrix

                's_fc': 0.0,            # scales influence of semantic similarity
                                        # on M_FC matrix.
                                        # s_fc is first implemented in
                                        # Healey et al. 2016;
                                        # set to 0.0 for prior papers.

                'phi_s': 4.661547054594787,      # primacy parameter
                'phi_d': 2.738934338758688,      # primacy parameter


                'epsilon_s': 0.0,      # baseline activiation for stopping probability 
                'epsilon_d': 2.723826426356652,        # scale parameter for stopping probability 

                'k':  5.380182482069175,        # scale parameter in luce choice rule during recall

                # parameters specific to optimal CMR:
                'primacy': 0.0,
                'enc_rate': 1.0,

            }


        # run probCMR on the data
        if N==0: # recall_mode = 0 to simulate based on parameters
            resp, times,_,reminders,recalls,accs,pcas = self.run_CMR2(
                recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                params=param_dict, subj_id_path=subjects_path, sep_files=False)   
            _,CMR_recalls,CMR_sp = self.data_recode(data_pres, resp)
            return CMR_sp,reminders,recalls,accs,pcas
        if N==1:# recall_mode = 1 to calculate likelihood of data based on parameters
            _, _,lkh,_,_,_,_ = self.run_CMR2(
                recall_mode=1,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                params=param_dict, subj_id_path=subjects_path, sep_files=False)   
            return lkh   
        else:
            resps = []
            for k in range(N):
                resp, times,_,reminders,recalls,accs,pcas = self.run_CMR2(
                    recall_mode=0,LSA_mat=LSA_mat, data_path=data_path, rec_path=data_rec_path,
                    params=param_dict, subj_id_path=subjects_path, sep_files=False)   
                _,CMR_recalls,CMR_sp = self.data_recode(data_pres, resp)
                resps.append(resp)

            
            RMSE, data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs = self.simulateCMR(resps, N, ll, lag_examine, LSA_mat, data_path, data_rec_path, subjects_path)
            self.plot_results(data_spc,CMR_spcs,data_pfr,CMR_pfrs,data_crp,CMR_crps,data_intrusion,CMR_intrusions,data_LSA,CMR_LSAs,data_sprob,CMR_sprobs)
            return -RMSE
