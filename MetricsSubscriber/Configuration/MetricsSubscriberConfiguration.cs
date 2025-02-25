
using System.Reflection;
using MassTransit;
using MetricsSubscriber.Consumers;
using MetricsSubscriber.Entity;
using MetricsSubscriber.Models.Bus;
using MetricsSubscriber.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace MetricsSubscriber.Configuration;

public class ModelPerformanceContext(DbContextOptions<ModelPerformanceContext> options) : DbContext(options)
{
	public DbSet<Metrics> Metrics { get; set; }
	public DbSet<ModelConfiguration> ModelConfigurations { get; set; }

	protected override void OnModelCreating(ModelBuilder modelBuilder)
	{
		base.OnModelCreating(modelBuilder);

		modelBuilder.Entity<Metrics>(entity =>
		{
			entity.HasKey(e => e.Id);
			entity.Property(e => e.Epoch).IsRequired();
			entity.Property(e => e.Loss).IsRequired();
			entity.Property(e => e.Accuracy).IsRequired();
			entity.Property(e => e.MeanPPV).IsRequired();
			entity.Property(e => e.MeanFPR).IsRequired();
			entity.Property(e => e.MeanRecall).IsRequired();
			entity.HasOne(e => e.ModelConfiguration)
				.WithOne()
				.HasForeignKey<ModelConfiguration>(e => e.Id)
				.OnDelete(DeleteBehavior.Cascade);
		});

		modelBuilder.Entity<ModelConfiguration>(entity =>
		{
			entity.HasKey(e => e.Id);
			entity.Property(e => e.Name).IsRequired();
			entity.Property(e => e.Layers).IsRequired();
			entity.Property(e => e.LearningRate).IsRequired();
			entity.Property(e => e.Hash).IsRequired();
		});
		// Hash is unique
		modelBuilder.Entity<ModelConfiguration>()
			.HasIndex(e => e.Hash)
			.IsUnique();
	}
}

public static class ServiceCollectionExtensions
{
	public static void AddEntityFramework(this IServiceCollection services, string connectionString)
	{
		services.AddDbContext<ModelPerformanceContext>();
	}

	public static void AddMassTransit(this IServiceCollection services)
	{
		services.AddSingleton<IMessageValidator<MetricsMessage>, MetricsValidator>();
		services.AddSingleton<IMessageTransformer<MetricsMessage, Metrics>, MetricsTransformer>();

		services.AddMassTransit(registrationContext =>
		{
			registrationContext.AddConsumers(Assembly.GetExecutingAssembly());
			RegisterBusExchangesAndQueues(registrationContext);
		});
	}

	private static void RegisterBusExchangesAndQueues(IBusRegistrationConfigurator registrationContext)
	{
		registrationContext.UsingRabbitMq((context, cfg) =>
		{
			var configuration = context.GetRequiredService<IConfiguration>();

			var section = configuration.GetSection("RabbitMq")
				?? throw new InvalidOperationException("RabbitMq section not found in configuration");
			var host = section.GetValue<string>("Host")
				?? throw new InvalidOperationException("RabbitMq Host not found in configuration");
			var username = section.GetValue<string>("Username")
				?? throw new InvalidOperationException("RabbitMq Username not found in configuration");
			var password = section.GetValue<string>("Password")
				?? throw new InvalidOperationException("RabbitMq Password not found in configuration");
			var virtualHost = section.GetValue<string>("VirtualHost") ?? "/";
			var timeSpan = section.GetValue<TimeSpan?>("RetryInterval") ?? TimeSpan.FromSeconds(5);

			cfg.Host(host, virtualHost, h =>
			{
				h.Username(username);
				h.Password(password);
			});

			cfg.UseRawJsonDeserializer(isDefault: true);
			cfg.UseRawJsonSerializer(isDefault: true);
			cfg.UseMessageRetry(r => r.Intervals(100, 500, 1000));
			cfg.UseDelayedRedelivery(r => r.Intervals(timeSpan));

			cfg.ReceiveEndpoint("metrics", e =>
			{
				e.ConfigureConsumer<MetricsConsumer>(context);
			});

			cfg.ReceiveEndpoint("model_configuration", e =>
			{
				e.ConfigureConsumer<ModelConfigurationConsumer>(context);
			});
		});
	}
}

