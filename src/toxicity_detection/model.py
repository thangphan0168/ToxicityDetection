import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModel


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output, ) = grad_outputs
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Allow dynamic adjustment of lambda during training"""
        self.lambda_ = lambda_


class CrossLingualToxicityDetector(nn.Module):
    def __init__(
        self,
        model_name="google/gemma-3-270m",
        num_languages=3,
        hidden_dropout_prob=0.1,
        grl_lambda=1.0,
        from_pretrained=False,
        dtype=torch.float
    ):
        """
        Cross-lingual toxicity detection model with GRL.
        
        Args:
            model_name: HuggingFace model identifier or config
            num_languages: Number of languages for language identification
            hidden_dropout_prob: Dropout probability for classification heads
            grl_lambda: Initial lambda value for gradient reversal
            from_pretrained: If True, load pretrained weights. If False, initialize randomly.
        """
        super(CrossLingualToxicityDetector, self).__init__()
        
        if from_pretrained:
            self.base_model = AutoModel.from_pretrained(model_name, dtype=dtype)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.base_model = AutoModel.from_config(config, dtype=dtype)
        
        self.hidden_size = self.base_model.config.hidden_size
        
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        # self.toxicity_head = nn.Sequential(
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(self.hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(512, 2)
        # )
        # self.language_head = nn.Sequential(
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(self.hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.Linear(512, num_languages)
        # )
        self.toxicity_head = nn.Linear(self.hidden_size, 2)
        self.language_head = nn.Linear(self.hidden_size, num_languages)
        
        self.num_languages = num_languages
    
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing toxicity and language logits
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Last token hidden state        
        pooled_output = outputs.last_hidden_state[:, -1, :]
        
        toxicity_logits = self.toxicity_head(pooled_output)
        reversed_features = self.grl(pooled_output)
        language_logits = self.language_head(reversed_features)
        
        result = {
            'toxicity_logits': toxicity_logits,
            'language_logits': language_logits
        }
        
        if return_features:
            result['features'] = pooled_output
        
        return result
    
    def predict_toxicity(self, input_ids, attention_mask):
        """Get toxic logits only"""
        outputs = self.forward(input_ids, attention_mask)
        return outputs['toxicity_logits']
    
    def set_grl_lambda(self, lambda_):
        """Update GRL lambda"""
        self.grl.set_lambda(lambda_)
    
    def load_base_model_weights(self, model_name_or_path):
        """
        Load pretrained weights into the base model after initialization.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
        """
        print(f"Loading pretrained weights from {model_name_or_path}...")
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.base_model.load_state_dict(pretrained_model.state_dict())
        print("✓ Pretrained weights loaded successfully!")
    
    @classmethod
    def from_pretrained_base(cls, model_name, num_languages=10, **kwargs):
        """
        Create model and load pretrained base model weights.
        
        Args:
            model_name: HuggingFace model identifier
            num_languages: Number of languages
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model with pretrained base weights
        """
        model = cls(
            model_name=model_name,
            num_languages=num_languages,
            from_pretrained=True,
            **kwargs
        )
        return model


if __name__ == "__main__":
    # Example: Create and test the model
    print("Creating Cross-Lingual Toxicity Detection Model...")
    
    # Method 1: Initialize with random weights (fast, no download)
    # print("\nMethod 1: Random initialization (no download)")
    # model = CrossLingualToxicityDetector(
    #     model_name="google/gemma-3-270m",
    #     num_languages=3,
    #     from_pretrained=False,
    #     dtype=torch.float
    # )
    # print(model.base_model.dtype)
    # print(f"✓ Model created with random weights")
    
    # Method 2: Initialize with pretrained weights (slower, downloads model)
    print("\nMethod 2: Pretrained initialization (downloads model)")
    model = CrossLingualToxicityDetector.from_pretrained_base(
        model_name="google/gemma-3-270m",
        num_languages=3
    )
    print(f"✓ Model created with pretrained weights")
    
    print(f"\nModel details:")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Number of languages: {model.num_languages}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"Toxicity logits shape: {outputs['toxicity_logits'].shape}")
    print(f"Language logits shape: {outputs['language_logits'].shape}")
    print("\nModel ready for training!")
