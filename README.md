# Bitcoin Predictor

A real-time Bitcoin transaction monitoring and analysis system that connects to Kraken's websocket API, processes transactions, and streams them to Kafka for further analysis.

## Features

- Real-time connection to Kraken's websocket API
- Transaction data streaming
- Kafka integration for data processing
- Scalable architecture for high-throughput transaction processing

## Prerequisites

- Python 3.8+
- Apache Kafka
- Kraken API credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/my_bitcoin_predictor.git
cd my_bitcoin_predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory with your Kraken API credentials:
```
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
```

2. Configure Kafka settings in `config.py`:
```python
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'bitcoin-transactions'
```

## Usage

1. Start the Kafka server:
```bash
# Make sure Kafka is running on your system
```

2. Run the main application:
```bash
python main.py
```

## Development Plan

1. Steps
    [ ] Create main file
        - Connect to Kraken socket
        - Pull transactions
        - Send transactions to kafka
            - Create topic
            - Connect topic

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kraken API for providing real-time cryptocurrency data
- Apache Kafka for distributed streaming platform