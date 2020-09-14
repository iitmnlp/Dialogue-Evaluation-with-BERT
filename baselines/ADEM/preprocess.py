import json
import codecs

class DataLoader(object):
	def __init__(self, fname, mode):
		self.fname = fname
		self.mode = mode

	def preprocess(self, line, speaker_tag='<first_speaker>'):
		line = speaker_tag + ' ' + line.lower()
		return line

	def load_data(self):

		dataset = []
		f = codecs.open(self.fname, 'r', encoding='utf-8')

		for line in f.readlines():
			df = json.loads(line.strip())
			c = ''
			context = df['context']
			for idx, turn in enumerate(context):
				speaker_tag = '<first_speaker>' if (idx%2==0) else '<second_speaker>'
				c += self.preprocess(turn, speaker_tag) + " "
			
			c = '</s> ' + c.strip() + ' </s>'
			r_gt = df['positive_responses'][0]

			speaker_tag = '<first_speaker>' if ((idx+1)%2==0) else '<second_speaker>'
			m_names = ['model_1', 'model_2','model_3', 'model_4','model_5', 'model_6','model_7', 'model_8']
			scores = [5]*4 + [1]*4
			
			m_rs = df['positive_responses'][1:5]

			if self.mode == 'random':
				m_rs += df['random_negative_responses'][:4]
			elif self.mode == 'adversarial':
				m_rs += df['adversarial_negative_responses'][:4]

			m_rs = ['</s> ' + self.preprocess(response, speaker_tag) + ' </s>' for response in m_rs]

			entry = { 'c': c, 'r_gt': r_gt, 'r_models': {}}
			for n, r, s in zip(m_names, m_rs, scores):
				entry['r_models'][n] = [r, s, len(r)]
			dataset.append(entry)
		
		f.close()
		
		return dataset

class Preprocessor(object):
	def preprocess(self, s):
		while '@@ ' in s:
			s = s.replace('@@ ', '')

		utterance = s.replace('@user', '<at>').replace('&lt;unk&gt;', '<unk>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')
		# Make sure we end with </s> token
		utterance = utterance.replace('user', '<at>')
		utterance = utterance.replace('A:', '<first_speaker>')
		utterance = utterance.replace('B:', '<second_speaker>')
		utterance = utterance.replace('& lt', '<')
		utterance = utterance.replace('& gt', '>')
		utterance = utterance.replace('&lt;', '<')
		utterance = utterance.replace('&gt;', '>')
		utterance = utterance.replace('\'', ' \'')
		utterance = utterance.replace('"', ' " ')
		utterance = utterance.replace("'", " '")
		utterance = utterance.replace(";", " ")
		utterance = utterance.replace("`", " ")
		utterance = utterance.replace("..", ".")
		utterance = utterance.replace("..", ".")
		utterance = utterance.replace("..", ".")
		utterance = utterance.replace(",,", ",")
		utterance = utterance.replace(",,", ",")
		utterance = utterance.replace(",,", ",")
		utterance = utterance.replace('.', ' . ')
		utterance = utterance.replace('!', ' ! ')
		utterance = utterance.replace('?', ' ? ')
		utterance = utterance.replace(',', ' , ')
		utterance = utterance.replace('~', '')
		utterance = utterance.replace('-', ' - ')
		utterance = utterance.replace('*', ' * ')
		utterance = utterance.replace('(', ' ')
		utterance = utterance.replace(')', ' ')
		utterance = utterance.replace('[', ' ')
		utterance = utterance.replace(']', ' ')
		utterance = re.sub('[\s]+', ' ', utterance)
		utterance = utterance.replace('  ', ' ')
		utterance = utterance.replace('  ', ' ')
		s = utterance
		while '! ! ! !' in s:
		    s = s.replace('! ! ! !', '! ! !')
		#s = utterance.replace('/', ' ')
		while s[-1] == ' ':
		    s = s[0:-1]
		if not s[-5:] == ' </s>':
		    s = s + ' </s>'
		return unicode(s)
