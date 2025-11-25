import pandas as pd
import pickle


class model_pipeline():
    
    def __init__(self):        
        self.container = None


        
        
class pipe_manager():
    
    
    def register(pipeline, pipe_path,  execution_id , inventory_path, model_ids= []):
        new_path =  str(pipe_path)+'/'+str(pipeline.id) + '.pkl'
        with open(new_path, 'wb') as outp:
            pickle.dump(pipeline, outp, pickle.HIGHEST_PROTOCOL)


        mdl_list = []
        if len(model_ids) > 0:
            for i in range(len(pipeline.model_list)):
                model_id = getattr(pipeline, pipeline.model_list[i]).id_
                if model_id in model_ids:
                    mdl_list.append([
                                 model_id,
                                 getattr(pipeline, pipeline.model_list[i]).creation_,
                                 int(getattr(pipeline, pipeline.model_list[i]).train_time_),
                                 pipeline.model_list[i],
                                 getattr(pipeline, pipeline.model_list[i]).train_gini,
                                 getattr(pipeline, pipeline.model_list[i]).cv_gini,
                                 getattr(pipeline, pipeline.model_list[i]).test_gini,
                                 pipeline.id,
                                 new_path,
                                 execution_id
                                 ])
        elif len(model_ids) == 0:
            for i in range(len(pipeline.model_list)):
                model_id = getattr(pipeline, pipeline.model_list[i]).id_
                mdl_list.append([
                             model_id,
                             getattr(pipeline, pipeline.model_list[i]).creation_,
                             int(getattr(pipeline, pipeline.model_list[i]).train_time_),
                             pipeline.model_list[i],
                             getattr(pipeline, pipeline.model_list[i]).train_gini,
                             getattr(pipeline, pipeline.model_list[i]).cv_gini,
                             getattr(pipeline, pipeline.model_list[i]).test_gini,
                             pipeline.id,
                             new_path,
                             execution_id
                             ])
            

        mdl_list = pd.DataFrame(mdl_list,columns=['Model_ID','Creation_Time','Train_Time','Model_Type','Train_Gini','CV_Gini','Test_Gini','Pipe_ID','Path','Execution_ID'])


        inv = pd.read_csv(inventory_path)    
        inv = inv.append(mdl_list)
        inv = inv.drop_duplicates(['Model_ID','Pipe_ID'])
        inv.to_csv(inventory_path,index=False)

        return inv
    
    
    def apply_pipe(test_df, model_id, inventory_path):
        
        inv = pd.read_csv(inventory_path) 
        model_path = inv[inv["Model_ID"]== model_id].iloc[0,:]["Path"]

        with open(str(model_path), 'rb') as f:
            pl = pickle.load(f)

        probs, preds = pl.apply(test_df,model_id)

        return probs, preds
    
    
    def best_model(pipe_list, sample='cv', metric='gini'):
        all_pipe_ids = []
        all_mdl_list = []
        all_mdl_ids = []
        all_train_ginis = []
        all_cv_ginis = []
        all_test_ginis = []
        for pipe in pipe_list:
            for p1 in pipe.model_list:
                all_pipe_ids.append(pipe.id)
                all_mdl_list.append(p1)
                all_mdl_ids.append(getattr(pipe, p1).id_)
                all_train_ginis.append(getattr(pipe, p1).train_gini)
                all_cv_ginis.append(getattr(pipe, p1).cv_gini)
                all_test_ginis.append(getattr(pipe, p1).test_gini)

        if sample == 'cv':
            gini_list = all_cv_ginis
        elif sample == 'train':
            gini_list = all_train_ginis
        elif sample == 'test':
            gini_list = all_test_ginis

        max_value = max(gini_list)
        max_index = gini_list.index(max_value)

        best_pipe_id = all_pipe_ids[max_index]
        best_mdl_id = all_mdl_ids[max_index]
        best_algo = all_mdl_list[max_index]
        best_metric = gini_list[max_index]

        print("Best pipe for "+str(sample) +' sample: '+ str(best_pipe_id))
        print("Best model for "+str(sample) +' sample: '+ str(best_mdl_id))
        print("Best model algorithm for "+str(sample) +' sample: '+ str(best_algo))
        print("Best " +str(metric)+ " for " + str(sample) +' sample: ' + str(best_metric))


        return best_pipe_id, best_mdl_id, best_algo, best_metric    
    