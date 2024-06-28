from django import forms


class CriptoForm(forms.Form):
    CRIPTOMONEDAS = [
        ('XRP', 'XRP'),
        ('ETH', 'Ethereum'),
        ('DOT', 'Polkadot'),
        ('SHIB', 'Shiba Inu'),
        ('XLM', 'Stellar'),
        ('LINK', 'Chainlink'),
        ('AVAX', 'Avalanche'),
        ('SOL', 'Solana'),
        ('MATIC', 'Polygon'),
        ('FET', 'Fetch.ai'),
        ('GRT', 'The Graph'),
        ('LTC', 'Litecoin'),
        ('UNI', 'Uniswap'),
        ('DOGE', 'Dogecoin'),
        ('NEAR', 'NEAR Protocol'),
        ('CHZ', 'Chiliz'),
        ('ADA', 'Cardano'),
        ('BCH', 'Bitcoin Cash'),
        ('APE', 'ApeCoin'),
    ]
    criptomoneda = forms.ChoiceField(choices=CRIPTOMONEDAS)
    fecha = forms.DateField(widget=forms.DateInput(attrs={'class': 'datepicker'}))
