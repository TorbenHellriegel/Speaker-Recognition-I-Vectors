import os
import yaml
import shutil
import sidekit
import numpy as np
from tqdm import tqdm
from utils import convert_wav, convert_wav_aug, safe_makedir, parse_yaml



class Initializer():
    """
    To build speaker verification model, one needs speech data from each speaker
    that is to be known by the system. The set of known speakers are in speaker
    recognition known as the (enrollment speakers), and a speaker is enrolled
    into the system when enrollment data from the speaker is processed to build
    its model.
    After the enrollment process, the performance of the speaker verification
    system can be evaluated using test data, which in an open set scenario, will
    consist of data from speakers in and outside the enrollment set.
    The set of all speakers involved in testing the system will be referred to
    as the test speakers.

    This class if for preprocessing and structure the preprocessed data
    into h5 files that will be used later for training and evaluating our models
    NOTE:All outputs of this script can be found in the directory self.task_dir
    """

    def __init__(self, conf_path):
        """
        This method parses the YAML configuration file which can be used for
        initializing the member varaibles!!
        Args:
            conf_path (String): path of the YAML configuration file
        """
        
        #location of output files
        self.conf = parse_yaml(conf_path)
        self.task_dir = os.path.join(self.conf['outpath'], "task")
        #location of audio files
        self.audio_dir = os.path.join(self.conf['outpath'], "audio")
        #location of all the audio data
        self.data_dir = os.path.join(self.audio_dir, "data")
        #location of just the enrollment audio data
        self.enroll_dir = os.path.join(self.audio_dir, "enroll")
        #location of just the test audio data
        self.test_dir = os.path.join(self.audio_dir, "test")


    def preprocess_audio(self):
        """
        Copy the Merged Arabic Corpus of Isolated Words into their
        associated directory. The whole audio data will be in 'data'
        directory, the enrolled data only will be in 'enroll', and the
        test data will be in 'test'.
        """
        #remove the data directory if exists
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        #remove the enroll directory if exists
        if os.path.exists(self.enroll_dir):
            shutil.rmtree(self.enroll_dir)
        #remove the test directory if exists
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        #create audio/enroll directory
        safe_makedir(self.enroll_dir)
        #create audio/test directory
        safe_makedir(self.test_dir)

        #iterate over speakers
        speakers = sorted(os.listdir(self.conf['inpath']))
        for sp in tqdm(speakers, desc="Converting Audio"):
            speaker_path = os.path.join(self.conf['inpath'], sp)
            sub_dir = os.listdir(speaker_path)
            wav_filenames = []
            for dir in sub_dir:
                wav_files = os.listdir(os.path.join(speaker_path, dir))
                for wav in wav_files:
                    wav_filenames.append(os.path.join(sp, dir, wav))
            for wav in wav_filenames:
                inwav = os.path.join(self.conf['inpath'], wav)
                outwav = os.path.join(self.data_dir, wav)
                outenroll = os.path.join(self.enroll_dir, wav)
                outwav0, outwav1, outwav2 = convert_wav_aug(inwav, outwav)
                # convert_wav(inwav,
                #             outwav,
                #             no_channels = self.conf['no_channels'],
                #             sampling_rate = self.conf['sampling_rate'],
                #             bit_precision = self.conf['bit_precision'])
                safe_makedir(os.path.split(outenroll)[0])
                outenroll0 = os.path.join(os.path.split(outenroll)[0],outwav0.split('/')[-1])
                outenroll1 = os.path.join(os.path.split(outenroll)[0],outwav1.split('/')[-1])
                outenroll2 = os.path.join(os.path.split(outenroll)[0],outwav2.split('/')[-1])
                shutil.copyfile(outwav0, outenroll0)
                shutil.copyfile(outwav1, outenroll1)
                shutil.copyfile(outwav2, outenroll2)

        #iterate over speakers
        speakers = sorted(os.listdir(os.path.join(os.path.split(self.conf['inpath'])[0],'vox1_test_wav')))
        for sp in tqdm(speakers, desc="Converting Audio"):
            speaker_path = os.path.join(os.path.join(os.path.split(self.conf['inpath'])[0],'vox1_test_wav'), sp)
            sub_dir = os.listdir(speaker_path)
            wav_filenames = []
            for dir in sub_dir:
                wav_files = os.listdir(os.path.join(speaker_path, dir))
                for wav in wav_files:
                    wav_filenames.append(os.path.join(sp, dir, wav))
            for wav in wav_filenames:
                inwav = os.path.join(os.path.join(os.path.split(self.conf['inpath'])[0],'vox1_test_wav'), wav)
                outwav = os.path.join(self.data_dir, wav)
                outtest = os.path.join(self.test_dir, wav)
                outwav0, outwav1, outwav2 = convert_wav_aug(inwav, outwav)
                # convert_wav(inwav,
                #             outwav,
                #             no_channels = self.conf['no_channels'],
                #             sampling_rate = self.conf['sampling_rate'],
                #             bit_precision = self.conf['bit_precision'])
                safe_makedir(os.path.split(outtest)[0])
                outtest0 = os.path.join(os.path.split(outtest)[0],outwav0.split('/')[-1])
                outtest1 = os.path.join(os.path.split(outtest)[0],outwav1.split('/')[-1])
                outtest2 = os.path.join(os.path.split(outtest)[0],outwav2.split('/')[-1])
                shutil.copyfile(outwav0, outtest0)
                shutil.copyfile(outwav1, outtest1)
                shutil.copyfile(outwav2, outtest2)


    def create_idMap(self, group):
        """
        IdMap are used to store two lists of strings and to map between them.
        Most of the time, IdMap are used to associate segments names (sessions)
        stored in leftids; with the ID of their class (that could be the speaker
        ID) stored in rightids.
        Additionally, and in order to allow more flexibility, IdMap includes two
        other vectors: 'start'and 'stop' which are float vectors used to store
        boudaries of audio segments.
        Args:
            group (string): name of the group that we want to create idmap for
        NOTE: Duplicated entries are allowed in each list.
        """
        assert group in ["enroll", "test"],\
            "Invalid group name!! Choose either 'enroll', 'test'"
        # Make enrollment (IdMap) file list
        group_dir = os.path.join(self.audio_dir, group)
        group_files = []
        for root, dirs, files in os.walk(group_dir):
            for file in files:
                #append the file name to the list
                group_files.append(os.path.join(os.path.split(os.path.split(root)[0])[1],os.path.split(root)[1],file))
        group_files = sorted(group_files)
        # list of model IDs
        group_models = [files.split('.')[0] for files in group_files]
        # list of audio segments IDs
        group_segments = [group+"/"+f for f in group_files]
        
        # Generate IdMap
        group_idmap = sidekit.IdMap()
        group_idmap.leftids = np.asarray(group_models)
        group_idmap.rightids = np.asarray(group_segments)
        group_idmap.start = np.empty(group_idmap.rightids.shape, '|O')
        group_idmap.stop = np.empty(group_idmap.rightids.shape, '|O')
        if group_idmap.validate():
            group_idmap.write(os.path.join(self.task_dir, group+'_idmap.h5'))
            #generate tv_idmap and plda_idmap as well
            if group == "enroll":
                group_idmap.write(os.path.join(self.task_dir, 'tv_idmap.h5'))
                group_idmap.write(os.path.join(self.task_dir, 'plda_idmap.h5'))
        else:
            raise RuntimeError('Problems with creating idMap file')


    def create_test_trials(self):
        """
        Ndx objects store trials index information, i.e., combination of 
        model and segment IDs that should be evaluated by the system which 
        will produce a score for those trials.

        The trialmask is a m-by-n matrix of boolean where m is the number of
        unique models and n is the number of unique segments. If trialmask(i,j)
        is true then the score between model i and segment j will be computed.
        """
        # Make list of test segments
        test_data_dir = os.path.join(self.audio_dir, "test") #test data directory
        test_files = []
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                #append the file name to the list
                test_files.append(os.path.join(os.path.split(os.path.split(root)[0])[1],os.path.split(root)[1],file))
        test_files = sorted(test_files)
        test_files = ["test/"+f for f in test_files]

        # Make lists for trial definition, and write to file
        test_models = []
        test_segments = []
        test_labels = []
        # Get enroll speakers
        enrolled_speakers = []
        for root, dirs, files in os.walk(os.path.join(self.audio_dir, "enroll")):
            for file in files:
                #append the file name to the list
                enrolled_speakers.append(os.path.join(os.path.split(os.path.split(root)[0])[1],os.path.split(root)[1],file).split('.')[0])
        enrolled_speakers = sorted(enrolled_speakers)
        for model in tqdm(enrolled_speakers, desc="Creating Test Cases"):
            for segment in sorted(test_files):
                test_model = segment.split(".")[0].split("/")[1]
                test_models.append(model)
                test_segments.append(segment)
                # Compare gender and speaker ID for each test file
                if test_model == model.split("/")[0]:
                    test_labels.append('target')
                else:
                    test_labels.append('nontarget')
            
        with open(os.path.join(self.task_dir, "test_trials.txt"), "w") as fh:
            for i in range(len(test_models)):
                fh.write(test_models[i]+' '+test_segments[i]+' '+test_labels[i]+'\n')


    def create_Ndx(self):
        """
        Key are used to store information about which trial is a target trial
        and which one is a non-target (or impostor) trial. tar(i,j) is true
        if the test between model i and segment j is target. non(i,j) is true
        if the test between model i and segment j is non-target.
        """
        #Define Key and Ndx from text file
        #SEE: https://projets-lium.univ-lemans.fr/sidekit/_modules/sidekit/bosaris/key.html
        key = sidekit.Key.read_txt(os.path.join(self.task_dir, "test_trials.txt"))
        ndx = key.to_ndx()
        if ndx.validate():
            ndx.write(os.path.join(self.task_dir, 'test_ndx.h5'))
        else:
            raise RuntimeError('Problems with creating idMap file')


    def structure(self):
        """
        This is the main method for this class, it calls all previous
        methods... that's basically what it does :)
        """
        self.preprocess_audio()
        self.create_idMap("enroll")
        self.create_idMap("test")
        self.create_test_trials()
        self.create_Ndx()

def data_inint_main():
    conf_filename = "conf.yaml"
    init = Initializer(conf_filename)
    init.structure()
    print("data_inint DONE!!")




if __name__ == "__main__":
    conf_filename = "conf.yaml"
    init = Initializer(conf_filename)
    init.structure()
    print("DONE!!")