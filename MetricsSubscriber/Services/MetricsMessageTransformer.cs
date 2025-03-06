using MetricsSubscriber.Entity;
using MetricsSubscriber.Models.Bus;

namespace MetricsSubscriber.Services;

public class MetricsTransformer : IMessageTransformer<MetricsMessage, Metrics>
{
	public Metrics Transform(MetricsMessage message)
	{
		var accuracy = message.Accuracy;
		if (accuracy > 1)
		{
			accuracy /= 100.0;
		}

		return new()
		{
			Epoch = message.Epoch,
			Loss = message.Loss,
			Accuracy = accuracy,
			MeanPPV = message.MeanPPV,
			MeanFPR = message.MeanFPR,
			MeanRecall = message.MeanRecall,
			ModelConfiguration = null,
		};
	}
}
