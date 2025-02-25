using MetricsSubscriber;
using MetricsSubscriber.Models.Bus;

namespace MetricsSubscriber.Services;

public class MetricsValidator : IMessageValidator<MetricsMessage>
{
	public bool Validate(ref readonly MetricsMessage? message)
	{
		return message switch
		{
			not null => message.Epoch >= 0
			&& message.Loss >= 0
			&& message.Accuracy >= 0
			&& message.Accuracy <= 100
			&& message.MeanPPV >= 0
			&& message.MeanPPV <= 1
			&& message.MeanFPR >= 0
			&& message.MeanFPR <= 1
			&& message.MeanRecall >= 0
			&& message.MeanRecall <= 1,
			_ => true,
		};

	}
}

