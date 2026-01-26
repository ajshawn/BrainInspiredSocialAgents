""" Multi-Agent Importance-weighted actor-learner architecture (IMPALA) agent."""

from marl.agents.impala.builder import IMPALABuilder
from marl.agents.impala.builder import PopArtIMPALABuilder
from marl.agents.impala.config import IMPALAConfig
from marl.agents.impala.learning import IMPALALearner
from marl.agents.impala.learning import IMPALALearnerME
from marl.agents.impala.learning import PopArtIMPALALearner
from marl.agents.impala.learning import PopArtIMPALALearnerME
from marl.agents.impala.networks import MultiHeadAttentionLayer
from marl.agents.impala.networks import make_network
from marl.agents.impala.networks import make_network_2
from marl.agents.impala.networks import make_network_attention
from marl.agents.impala.networks import make_network_attention_tanh
from marl.agents.impala.networks import make_network_attention_spatial
from marl.agents.impala.networks import make_network_attention_item_aware
from marl.agents.impala.networks import make_network_attention_multihead
from marl.agents.impala.networks import make_network_attention_multihead_ff
from marl.agents.impala.networks import make_network_attention_multihead_disturb
from marl.agents.impala.networks import make_network_attention_multihead_enhance
from marl.agents.impala.networks import make_network_attention_multihead_item_aware
from marl.agents.impala.networks import make_network_impala_cnn_visualization
from marl.agents.impala.networks import make_network_attention_multihead_self_supervision
from marl.agents.impala.simpletr import make_network_simple_transformer
from marl.agents.impala.simpletr import make_network_transformer_attention
from marl.agents.impala.simpletr import make_network_transformer_cnnfeedback