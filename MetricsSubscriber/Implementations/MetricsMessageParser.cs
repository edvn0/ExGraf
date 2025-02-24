using MetricsSubscriber.Models.Vertical;

namespace MetricsSubscriber.Implementations;

public class MetricsMessageParser : IMessageParser<MetricsUpdatedEvent>
{
	MetricsUpdatedEvent? IMessageParser<MetricsUpdatedEvent>.TryParse(string message)
	{
		try
		{
			var values = message.Split(',').Select(double.Parse).ToArray();
			if (values.Length != 6) return null;

			var notification = new MetricsUpdatedEvent(
				values[0], values[1], values[2], values[3], values[4], values[5]
			);
			return notification;
		}
		catch (Exception)
		{
			return null;
		}
	}
}
